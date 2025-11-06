# region imports
from datetime import time, timedelta
from AlgorithmImports import *
from QuantConnect.Data.Consolidators import TickConsolidator
from QuantConnect.Indicators import ExponentialMovingAverage
import pytz
# endregion

class MesGoldenDeathCrossRTH(QCAlgorithm):
    """
    Trades 1 contract of Micro E-mini S&P 500 (MES) on a simple
    20 / 25 EMA cross built on 1000-tick bars, but **only** during
    09 : 30 → 16 : 00 US-Eastern.  Any open position is closed no later
    than 15 : 58 ET and a $100 stop-loss is attached while a trade is open.
    """

    FAST, SLOW = 20, 25
    CONTRACTS  = 1
    STOP_RISK  = 100          # $ stop for the WHOLE position
    POINT_VAL  = 5.0          # $ per index-point for one MES contract

    TZ_NY      = pytz.timezone("America/New_York")
    OPEN_ET    = time(9, 30)
    CLOSE_ET   = time(16, 0)
    FLAT_ET    = time(15, 58)  # hard-flatten deadline
    SYMBOL     = Futures.Indices.SP_500_E_MINI   # CME - "ES"

    # ---------------- initialise -----------------
    def initialize(self):
        self.set_start_date(2025, 9, 1)
        self.set_end_date  (2025, 11, 3)
        self.set_cash(40_000)

        self.universe_settings.resolution = Resolution.TICK

        fut = self.add_future(self.SYMBOL, Resolution.TICK,
                              leverage = 1,
                              extended_market_hours = True,
                              data_normalization_mode=DataNormalizationMode.RAW)
        fut.set_filter(timedelta(0), timedelta(days=90))

        self.contract   = None
        self.indicators = {}          # {Symbol: {'fast','slow'}}
        self.tick_consolidators = {}
        self.latest_bar = {}
        self.last_bar_time = {}
        self.prev_diff  = 0.0
        self.stop_ticket = None

    # ---------------- indicator helper -----------
    def _ensure(self, symbol: Symbol):
        if symbol in self.indicators:
            return

        consolidator = TickConsolidator(1000)
        consolidator.DataConsolidated += lambda _, bar: self._on_consolidated(symbol, bar)
        self.subscription_manager.add_consolidator(symbol, consolidator)

        fast = ExponentialMovingAverage(f"{symbol.value}_fast", self.FAST)
        slow = ExponentialMovingAverage(f"{symbol.value}_slow", self.SLOW)

        self.register_indicator(symbol, fast, consolidator, lambda bar: bar.close)
        self.register_indicator(symbol, slow, consolidator, lambda bar: bar.open)

        self.indicators[symbol] = {"fast": fast, "slow": slow}
        self.tick_consolidators[symbol] = consolidator
        self.last_bar_time[symbol] = None

    # ---------------- on_data --------------------
    def on_data(self, slice: Slice):

        # pick front contract each day
        if slice.futures_chains:
            for chain in slice.futures_chains.values():
                if not chain.Contracts:
                    continue
                front = sorted(chain.Contracts.values(), key=lambda c: c.Expiry)[0]
                if self.contract != front.Symbol:
                    self._cancel_stop()
                    if self.contract:
                        consolidator = self.tick_consolidators.pop(self.contract, None)
                        if consolidator:
                            self.subscription_manager.remove_consolidator(self.contract, consolidator)
                        self.indicators.pop(self.contract, None)
                        self.latest_bar.pop(self.contract, None)
                        self.last_bar_time.pop(self.contract, None)
                    self.contract = front.Symbol
                    self.prev_diff = 0.0
                    self._ensure(self.contract)
                    self.latest_bar.pop(self.contract, None)
                    self.last_bar_time.pop(self.contract, None)
                break

        if self.contract is None:
            return

        self._ensure(self.contract)
        bar = self.latest_bar.get(self.contract)
        if bar is None:
            return

        end_time = bar.EndTime
        if self.last_bar_time.get(self.contract) == end_time:
            return
        self.last_bar_time[self.contract] = end_time

        ind = self.indicators[self.contract]
        if not (ind["fast"].is_ready and ind["slow"].is_ready):
            return

        # current ET
        t_et = self.time.astimezone(self.TZ_NY).time()

        # force-flat 15:58 or later
        if t_et >= self.FLAT_ET:
            if self.portfolio.invested:
                self._cancel_stop()
                self.liquidate(self.contract)
            return

        # only trade in RTH window
        if not (self.OPEN_ET <= t_et < self.CLOSE_ET):
            return

        # ---------------- EMA cross ----------------
        diff = ind["fast"].Current.Value - ind["slow"].Current.Value
        cross_up   = self.prev_diff <= 0 < diff
        cross_down = self.prev_diff >= 0 > diff
        self.prev_diff = diff

        qty = self.portfolio[self.contract].quantity
        # ---------------- exits --------------------
        if qty > 0 and cross_down:
            self._cancel_stop()
            self.liquidate(self.contract)
            qty = 0
        elif qty < 0 and cross_up:
            self._cancel_stop()
            self.liquidate(self.contract)
            qty = 0

        # ---------------- entries -----------------
        if qty == 0:
            points_risk = self.STOP_RISK / (self.CONTRACTS * self.POINT_VAL)

            if cross_up:
                entry = bar.Close
                stop  = round(entry - points_risk, 2)
                self.market_order(self.contract,  self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract, -self.CONTRACTS, stop, tag="SL-$100 LONG")

            elif cross_down:
                entry = bar.Close
                stop  = round(entry + points_risk, 2)
                self.market_order(self.contract, -self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract,  self.CONTRACTS, stop, tag="SL-$100 SHORT")

    # -------------- util helpers ----------------
    def _on_consolidated(self, symbol: Symbol, bar: TradeBar):
        self.latest_bar[symbol] = bar

    def _cancel_stop(self):
        if self.stop_ticket and self.stop_ticket.status in (OrderStatus.NEW,
                                                            OrderStatus.SUBMITTED,
                                                            OrderStatus.PARTIALLY_FILLED):
            self.stop_ticket.cancel()
        self.stop_ticket = None

    def on_end_of_algorithm(self):
        self._cancel_stop()
        for symbol, consolidator in list(self.tick_consolidators.items()):
            self.subscription_manager.remove_consolidator(symbol, consolidator)
        self.tick_consolidators.clear()
