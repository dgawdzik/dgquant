# region imports
from AlgorithmImports import * 
import csv
# endregion

class Alg(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2014, 6, 5)  # Set Start Date
        self.set_end_date(2014, 6, 5)  # Set End Date
        self.set_cash(100000)  # Set Strategy Cash
        
        # Market hours filter - using milliseconds since midnight in EST timezone
        # 9:30 AM EST = 9 hours * 60 min * 60 sec * 1000 ms + 30 min * 60 sec * 1000 ms = 34,200,000 ms
        # 4:00 PM EST = 16 hours * 60 min * 60 sec * 1000 ms = 57,600,000 ms
        self.market_open_milliseconds = 34200000   # 9:30 AM EST in milliseconds
        self.market_close_milliseconds = 57600000  # 4:00 PM EST in milliseconds
                
        self.symbol = self.add_equity("AAPL", Resolution.MINUTE, data_normalization_mode=DataNormalizationMode.Raw).Symbol
        
        # Add EMA indicators
        self.ema10 = self.ema(self.symbol, 10, Resolution.MINUTE)
        self.ema30 = self.ema(self.symbol, 30, Resolution.MINUTE)
        
        # Ensure indicators are warmed up before trading
        self.set_warm_up(31)
        self.is_buying = False
        self.bar_count = 0
        
        # We store bar data (time, open, high, low, close, volume) in a list
        self.price_data = []
        
        # We store trades in a list for later analysis
        self.trades = []
    
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
                self.ema10.Current.Value,
                self.ema30.Current.Value
            ))
        
        # Skip if we're still in the warm-up period
        if self.is_warming_up:
            return
            
        # Skip if our symbol isn't in the data
        if not self.symbol in data:
            return
        
        # Get current time in milliseconds since midnight in EST
        # current_time = self.Time.time()
        # current_milliseconds = (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) * 1000
        
        # # Skip if outside regular market hours
        # if current_milliseconds < self.market_open_milliseconds or current_milliseconds > self.market_close_milliseconds:
        #     return
        
        holdings = self.Portfolio[self.symbol].quantity

        # Check for buy signal: EMA10 crosses above EMA30
        if not self.is_buying and self.ema10.current.value > self.ema30.Current.Value and holdings <= 0:           
            if holdings < 0:
                self.liquidate(self.symbol)
                self.debug(f"Cover signal: EMA10 ({self.ema10.Current.Value:.2f}) crossed above EMA30 ({self.ema30.Current.Value:.2f})")
            # Go long
            self.set_holdings(self.symbol, 1)
            self.debug(f"Buy signal: EMA10 ({self.ema10.Current.Value:.2f}) crossed above EMA30 ({self.ema30.Current.Value:.2f})")
            self.is_buying = True

        # Check for sell signal: EMA10 crosses below EMA30
        elif self.is_buying and self.ema10.Current.Value < self.ema30.Current.Value and holdings > 0:
            self.liquidate(self.symbol)
            self.debug(f"Sell signal: EMA10 ({self.ema10.Current.Value:.2f}) crossed below EMA30 ({self.ema30.Current.Value:.2f})")
            # sell short
            self.set_holdings(self.symbol, -1)
            self.debug(f"Sell short signal: EMA10 ({self.ema10.Current.Value:.2f}) crossed below EMA30 ({self.ema30.Current.Value:.2f})")
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
            writer.writerow(["time", "open", "high", "low", "close", "volume", "ema10", "ema30"])
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
