# region imports
from AlgorithmImports import *
import datetime
# endregion

class DgUniverseSelectionSmallCaps(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 9, 29)
        self.set_end_date(2025, 10,1)

        self.set_cash(30000)

        self.universe_settings.minimum_time_in_universe = datetime.timedelta(days = 0)
        self.universe_settings.extended_market_hours = True
        self.add_universe(self.select_universe)

        self.symbols = []
        self.symbol_to_macd = {}
        self.symbol_to_consolidator = {}
        self.is_warm_up = False
        self.in_trading_time_interval = False

        # Symbol outside of universe selection, used for defining trading time interval, and market hours for the exchange
        self.nvda_symbol: Symbol = self.add_equity(ticker= "NVDA", resolution = Resolution.MINUTE).symbol
        
        self.schedule.on(
            date_rule = self.date_rules.every_day(symbol = self.nvda_symbol),
            time_rule = self.time_rules.at(hour = 4, minute = 0, second = 0), # Start trading on pre-market open 
            callback = self.set_start_trading
            )

        self.schedule.on(
            date_rule = self.date_rules.every_day(symbol = self.nvda_symbol),
            time_rule = self.time_rules.before_market_open(symbol = self.nvda_symbol, minutes_before_open = 5), # Stop trading 5 min before regural market open
            callback = self.set_stop_trading
            )

    def set_start_trading(self):
        self.in_trading_time_interval = True

    def set_stop_trading(self):
        self.in_trading_time_interval = False

    def select_universe(self, fundamentals : list[Fundamental]) -> list[Symbol]:
        result = []

        for fundamental in fundamentals:
            if fundamental.price > 5 and fundamental.earning_reports.basic_eps.value > 1:
                # Debt to Equity ration = total debt / total equity
                if fundamental.financial_statements.balance_sheet.total_debt.value / fundamental.financial_statements.balance_sheet.total_equity.value > 1:
                    if fundamental.operation_ratios.roe.value > 0.2:
                        if fundamental.operation_ratios.revenue_growth.one_year > 0.1:
                            if fundamental.operation_ratios.net_income_growth.one_year > 0.1:
                                #if fundamental.asset_classification.morning_star_industry_code == MorningstarIndustryCode.SEMICONDUCTORS:
                                if fundamental.symbol != self.nvda_symbol:
                                    result.append(fundamental.symbol)
                                    # self.log(fundamental.symbol.value)

        # Add open order symbols to universe
        for symbol in self.symbols:
            symbol_open_order_tickets = list(self.transactions.get_open_order_tickets(symbol=symbol))
            if (self.portfolio[symbol].invested or len(symbol_open_order_tickets) > 0) and symbol not in result:
                result.append(symbol)

        return result

    def on_securities_changed(self, changes: SecurityChanges) -> None:
        self.is_warm_up = False
        
        added_securities = []

        for security in changes.added_securities:
            if security.symbol != self.nvda_symbol:
                added_securities.append(security.symbol)

        # temp MACD so we can compute how much historical data is needed
        temp_macd = MovingAverageConvergenceDivergence(fast_period = 12, slow_period = 26, signal_period = 9, type = MovingAverageType.EXPONENTIAL)
        macd_warm_up_period = temp_macd.warm_up_period * 5 # Account for consolidator for 5 min
        history = self.history(tickers = added_securities, periods = macd_warm_up_period, resolution = Resolution.MINUTE)

        for symbol in added_securities:
            self.symbols.append(symbol)
            self.symbol_to_macd[symbol] = MovingAverageConvergenceDivergence(fast_period = 12, slow_period = 26, signal_period = 9)
            
            self.symbol_to_consolidator[symbol] = TradeBarConsolidator(period = datetime.timedelta(minutes = 5))
            self.symbol_to_consolidator[symbol].data_consolidated += self.on_data_consolidated_5_min

            self.register_indicator(symbol = symbol, indicator = self.symbol_to_macd[symbol], consolidator = self.symbol_to_consolidator[symbol])
            
            try:
                # Warm up MACD for each symbol
                symbol_history = history.loc[symbol]
                for time, row in symbol_history.iterrows():
                    # self.symbol_to_macd[symbol].update(time, row.close) without consolidator
                    trade_bar = TradeBar(time = time - datetime.timedelta(minutes = 1), symbol = symbol, open = row.open, high = row.high, low = row.low, close = row.close, volume = row.volume, period = datetime.timedelta(minutes = 1))
                    self.symbol_to_consolidator[symbol].update(trade_bar)
                
                self.log(f"{self.time}: Added {symbol}")
            except:
                self.log(f"{self.time}: History missing for {symbol}")

                self.symbols.remove(symbol)
                self.unregister_indicator(indicator = self.symbol_to_macd[symbol])
                self.symbol_to_macd.pop(symbol)
                self.subscription_manager.remove_consolidator(symbol = symbol, consolidator = self.symbol_to_consolidator[symbol])
                self.symbol_to_consolidator.pop(symbol)

        for security in changes.removed_securities:
            if security.symbol in self.symbols:
                self.symbols.remove(security.symbol)
                self.deregister_indicator(indicator = self.symbol_to_macd[security.symbol])
                self.symbol_to_macd.pop(security.symbol)
                self.log(f"{self.time}: Removed {security.symbol}")
                self.subscription_manager.remove_consolidator(symbol = security.symbol, consolidator = self.symbol_to_consolidator[security.symbol])
                self.symbol_to_consolidator.pop(security.symbol)

        self.is_warm_up = True
    
    def on_data_consolidated_5_min(self, sender, bar):
        if self.is_warm_up and self.in_trading_time_interval:
            symbol = bar.symbol
            symbol_open_order_tickets = list(self.transactions.get_open_order_tickets(symbol=symbol))

            if len(symbol_open_order_tickets) == 0 and self.symbol_to_macd[symbol].is_ready:
                if self.symbol_to_macd[symbol].current.value > self.symbol_to_macd[symbol].signal.current.value:
                    if not self.portfolio[symbol].invested:
                        self.limit_order(symbol=symbol, quantity=10, limit_price=bar.close)
                elif self.symbol_to_macd[symbol].current.value < self.symbol_to_macd[symbol].signal.current.value:
                    if self.portfolio[symbol].invested:
                        self.limit_order(symbol=symbol, quantity = -self.portfolio[symbol].quantity, limit_price=bar.close)

    def on_data(self, data: Slice):
        pass