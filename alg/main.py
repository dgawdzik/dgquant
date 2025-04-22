# region imports
from AlgorithmImports import * 
import csv
# endregion

class Alg(QCAlgorithm):
    
    FAST_EMA_PERIOD: int = 23
    SLOW_EMA_PERIOD: int = 50
    SYMBOL: str = "NVDA"
    
    def initialize(self):
        self.set_start_date(2025, 4, 16)  # Set Start Date
        self.set_end_date(2025, 4, 16)  # Set End Date
        self.set_cash(10_000)  # Set Strategy Cash
        
        self.universe_settings.resolution             = Resolution.MINUTE
        self.universe_settings.extended_market_hours  = True
        
        # Market hours filter - using milliseconds since midnight in EST timezone
        # 9:30 AM EST = 9 hours * 60 min * 60 sec * 1000 ms + 30 min * 60 sec * 1000 ms = 34,200,000 ms
        # 4:00 PM EST = 16 hours * 60 min * 60 sec * 1000 ms = 57,600,000 ms
        self.market_open_milliseconds = 34200000   # 9:30 AM EST in milliseconds
        self.market_close_milliseconds = 57600000  # 4:00 PM EST in milliseconds
                
        self.symbol = self.add_equity(
            ticker = Alg.SYMBOL,
            resolution = Resolution.MINUTE,
            market = Market.USA,
            fill_forward = True,
            leverage = 4,
            extended_market_hours = True, 
            data_normalization_mode = DataNormalizationMode.Raw
            ).Symbol
    
        # Add EMA indicators
        self.ema_fast = self.ema(self.symbol, Alg.FAST_EMA_PERIOD, Resolution.MINUTE)
        self.ema_slow = self.ema(self.symbol, Alg.SLOW_EMA_PERIOD, Resolution.MINUTE)
        
        # Ensure indicators are warmed up before trading
        self.set_warm_up(Alg.SLOW_EMA_PERIOD)
        self.is_buying = False
        self.bar_count = 0
        
        # We store bar data (time, open, high, low, close, volume) in a list
        self.price_data = []
        
        # We store trades in a list for later analysis
        self.trades = []
        self.prev_ema_diff = 0
    
    def on_data(self, data: Slice):
        """on_data event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        """
        self.bar_count += 1
        bar = data.bars.get(self.symbol)
        if bar is not None:
            # Capture the OHLC and Volume
            # self.Time is the current bar's end time
            self.price_data.append((
                self.Time,          # Bar end time (C# DateTime)
                bar.Open,
                bar.High,
                bar.Low,
                bar.Close,
                bar.Volume,
                self.ema_fast.Current.Value,
                self.ema_slow.Current.Value
            ))
        
        # Skip if we're still in the warm-up period
        if self.is_warming_up:
            return
        
        curr_ema_diff = self.ema_fast.Current.Value - self.ema_slow.Current.Value
        is_crossing = (self.prev_ema_diff < 0 and curr_ema_diff >= 0) or (self.prev_ema_diff > 0 and curr_ema_diff <= 0)
        self.prev_ema_diff = curr_ema_diff
        
        # Skip if our symbol isn't in the data
        if not self.symbol in data or not is_crossing:
            self.debug(f"Skipping trading logic, symbol {self.symbol} not in data or no crossing detected.")
            self.debug(f"Current EMA Fast: {self.ema_fast.Current.Value}, EMA Slow: {self.ema_slow.Current.Value}")
            self.debug(f"Previous EMA Diff: {self.prev_ema_diff}, Current EMA Diff: {curr_ema_diff}")
            return
        
        # Get current time in milliseconds since midnight in EST
        # current_time = self.Time.time()
        # current_milliseconds = (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) * 1000
        
        # # Skip if outside regular market hours
        # if current_milliseconds < self.market_open_milliseconds or current_milliseconds > self.market_close_milliseconds:
        #     return
        
        holdings = self.Portfolio[self.symbol].quantity

        # Check for buy signal: fast EMA crosses above slow EMA
        if curr_ema_diff > 0:
            if holdings < 0:
                self.liquidate(self.symbol)
                self.debug(f"Cover signal: fast EMA_{Alg.FAST_EMA_PERIOD} ({self.ema_fast.Current.Value:.2f}) crossed above slow EMA_{Alg.SLOW_EMA_PERIOD} ({self.ema_slow.Current.Value:.2f})")
            # Go long
            self.set_holdings(self.symbol, 1)
            self.debug(f"Buy signal: fast EMA_{Alg.FAST_EMA_PERIOD} ({self.ema_fast.Current.Value:.2f}) crossed above slow EMA_{Alg.SLOW_EMA_PERIOD} ({self.ema_slow.Current.Value:.2f})")
            self.is_buying = True

        # Check for sell signal: fast EMA crosses below slow EMA
        else:
            if holdings > 0:
                self.liquidate(self.symbol)
                self.debug(f"Sell signal: fast EMA_{Alg.FAST_EMA_PERIOD} ({self.ema_fast.Current.Value:.2f}) crossed below slow EMA_{Alg.SLOW_EMA_PERIOD} ({self.ema_slow.Current.Value:.2f})")
            # sell short
            self.set_holdings(self.symbol, -1)
            self.debug(f"Sell short signal: fast EMA_{Alg.FAST_EMA_PERIOD} ({self.ema_fast.Current.Value:.2f}) crossed below slow EMA_{Alg.SLOW_EMA_PERIOD} ({self.ema_slow.Current.Value:.2f})")
            self.is_buying = False
    
    def on_order_event(self, orderEvent: OrderEvent):
        if orderEvent.status == OrderStatus.Filled:
            fill_time = self.Time  # or orderEvent.UtcTime if you want precise fill time
            fill_price = orderEvent.fill_price
            fill_quantity = orderEvent.fill_quantity
            order = str(self.transactions.get_order_by_id(orderEvent.order_id))
            side = "Buy" if fill_quantity > 0 else "Sell"

            account_balance = self.Portfolio.TotalPortfolioValue
            self.trades.append((
                fill_time,
                orderEvent.Symbol.Value,
                fill_price,
                fill_quantity,
                account_balance,
                order,
                side
            ))
            self.debug(f"on_order_event >> {orderEvent.Symbol.Value}" +
                        f" filled {fill_quantity} @ {fill_price} ({str(order)}).")

    def on_end_of_algorithm(self):
        self.debug(f"Total number of 1m data points received: {self.bar_count}")
        
        filename = "/Lean/Data/price_data.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["time", "open", "high", "low", "close", "volume", "ema_fast", "ema_slow"])
            # Data rows
            for row in self.price_data:
                # row[0] is a DateTime; convert it to ISO string or just str(row[0])
                writer.writerow([
                    row[0].strftime("%Y-%m-%d %H:%M:%S"),
                    row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                ])
        self.debug(f"Wrote {len(self.price_data)} bars to {filename}")
        
        # Write trades data to trades.csv
        trades_filename = "/Lean/Data/trades.csv"
        with open(trades_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "symbol", "price", "quantity", "account_balance", "order_type", "side"])
            for trade in self.trades:
                writer.writerow(trade)
        self.debug(f"Wrote {len(self.trades)} trades to {trades_filename}")
        
        return super().on_end_of_algorithm()
