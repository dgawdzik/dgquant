from AlgorithmImports import *
from datetime import timedelta

class AlgMulti(QCAlgorithm):

    def initialize(self) -> None:
        # Backtest period and cash
        self.set_start_date(2025, 5, 1)  
        self.set_end_date(2025, 5, 1)  
        self.set_cash(40_000)  
        # Request S&P 500 E-mini futures chain
        future_chain = self.add_future(
            Futures.Indices.SP_500_E_MINI,
            resolution=Resolution.MINUTE
        )

        future_chain.set_filter(timedelta(0), timedelta(365))

        # 2) Immediately resolve the front contract
        all_contracts = self.future_chain_provider.get_future_contract_list(
            future_chain.symbol, self.start_date
        )

        # pick the nearest expiry
        front = sorted(all_contracts, key=lambda x: x.ID.Date)[0]

        # 3) subscribe that single contract by symbol
        sub = self.add_future_contract(front, Resolution.MINUTE)
        self.contract_symbol = sub.symbol
        self.debug(f"Selected contract: {self.contract_symbol}")

        # --- 30-minute consolidator & indicators ---
        consolidator30 = TradeBarConsolidator(timedelta(minutes=30))
        self.ema23_30 = ExponentialMovingAverage(23)
        self.ema50_30 = ExponentialMovingAverage(50)
        self.macd_30 = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.EXPONENTIAL)
        
        # Register consolidated data and attach indicators
        self.consolidate(self.symbol, consolidator30)
        self.register_indicator(self.symbol, consolidator30, self.ema23_30)
        self.register_indicator(self.symbol, consolidator30, self.ema50_30)
        self.register_indicator(self.symbol, consolidator30, self.macd_30)
        
        # --- 5-minute consolidator & indicators ---
        consolidator5 = TradeBarConsolidator(timedelta(minutes=5))
        self.ema23_5 = ExponentialMovingAverage(23)
        self.ema50_5 = ExponentialMovingAverage(50)
        self.macd_5 = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.EXPONENTIAL)
        
        # Register consolidated data and attach indicators
        self.consolidate(self.symbol, consolidator5)
        self.register_indicator(self.symbol, consolidator5, self.ema23_5)
        self.register_indicator(self.symbol, consolidator5, self.ema50_5)
        self.register_indicator(self.symbol, consolidator5, self.macd_5)
        
        # 4) indicators on the bound symbol
        self.ema23 = self.EMA(self.contract_symbol, 23, Resolution.MINUTE)
        self.ema50 = self.EMA(self.contract_symbol, 50, Resolution.MINUTE)
      
        self.is_long = False
        self.is_short = False
        self.bar_count = 0

        
    def on_data(self, slice: Slice) -> None:
        # 3) Wait for indicators to warm up
        if not (self.ema23.is_ready and self.ema50.is_ready and self.macd_30.is_ready and self.macd_5.is_ready and self.ema23_30.is_ready and self.ema50_30.is_ready and self.ema23_5.is_ready and self.ema50_5.is_ready):
            return

        # 4) Ensure we have a bar for the selected contract
        bar = slice.bars.get(self.contract_symbol)
        if bar is None:
            return

        # At 15:58 local time → exit
        if self.time.hour == 15 and self.time.minute >= 58:
            if self.portfolio[self.contract_symbol].invested:
                self.liquidate(self.contract_symbol)
                self.debug(f"Hard-exit 2 min before close at {str(self.time)}")
                return

        self.bar_count += 1
        ema10_val = self.ema23.current.value
        ema30_val = self.ema50.current.value

        self.plot("EMA", "10", ema10_val)
        self.plot("EMA", "30", ema30_val)
        
        # 5) Go long on EMA23 > EMA50
        if not self.is_long and ema10_val > ema30_val:
            if self.portfolio[self.contract_symbol].invested:
                self.liquidate(self.contract_symbol)
                self.debug(f"Cover @ {self.time}  EMA10 {ema10_val:.2f} > EMA30 {ema30_val:.2f}")
            self.set_holdings(self.contract_symbol, 1)
            self.debug(f"Long @ {self.time}  EMA10 {ema10_val:.2f} > EMA30 {ema30_val:.2f}")
            self.is_long = True
            self.is_short = False

        # 6) Flip to short on EMA10 < EMA30
        elif not self.is_short and ema10_val < ema30_val:
            if self.portfolio[self.contract_symbol].invested:
                self.liquidate(self.contract_symbol)
                self.debug(f"Exit Long @ {self.time}  EMA10 {ema10_val:.2f} < EMA30 {ema30_val:.2f}")
            self.set_holdings(self.contract_symbol, -1)
            self.debug(f"Short @ {self.time}  EMA10 {ema10_val:.2f} < EMA30 {ema30_val:.2f}")
            self.is_long = False
            self.is_short = True

    def on_end_of_algorithm(self) -> None:
        self.debug(f"Total minute bars processed: {self.bar_count}")
        super().on_end_of_algorithm()
