# region imports
from datetime import timedelta
import time
from AlgorithmImports import *
import pytz
# endregion

class MesGoldenDeathCrossRTH(QCAlgorithm):
    """
    Trades 2 contracts of Micro E-mini S&P 500 (MES) on a simple
    23 / 50-EMA cross, but **only** during
    09 : 30 → 16 : 00 US-Eastern.  Any open position is closed no later
    than 15 : 58 ET and a $100 stop-loss is attached while a trade is open.
    """

    FAST, SLOW = 23, 50
    CONTRACTS  = 2
    STOP_RISK  = 100          # $ stop for the WHOLE position
    POINT_VAL  = 5.0          # $ per index-point for one MES contract

    TZ_NY      = pytz.timezone("America/New_York")
    OPEN_ET    = time(9, 30)
    CLOSE_ET   = time(16, 0)
    FLAT_ET    = time(15, 58)  # hard-flatten deadline
    SYMBOL     = Futures.Indices.SP_500_E_MINI   # CME - "ES"

    RSI_LEN    = 60
    RSI_OB     = 70           # over-bought  → short only above this
    RSI_OS     = 30           # over-sold    → long  only below this

    # ---------------- initialise -----------------
    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date  (2025, 4, 25)
        self.set_cash(40_000)

        self.universe_settings.resolution = Resolution.MINUTE

        fut = self.add_future(self.SYMBOL, Resolution.MINUTE,
                              leverage = 1,
                              extended_market_hours = True,
                              data_normalization_mode=DataNormalizationMode.RAW)
        fut.set_filter(timedelta(0), timedelta(days=90))

        self.contract   = None
        self.indicators = {}          # {Symbol: {'fast','slow','rsi'}}
        self.prev_diff  = 0.0
        self.stop_ticket = None

    # ---------------- indicator helper -----------
    def _ensure(self, symbol: Symbol):
        if symbol in self.indicators:
            return

        fast = self.ema(symbol, self.FAST, Resolution.MINUTE)
        slow = self.ema(symbol, self.SLOW, Resolution.MINUTE)
        rsi = self.rsi(symbol, self.RSI_LEN, MovingAverageType.SIMPLE, Resolution.MINUTE)

        self.indicators[symbol] = {"fast": fast, "slow": slow, "rsi": rsi}

    # ---------------- on_data --------------------
    def on_data(self, slice: Slice):

        # pick front contract each day
        if self.contract is None or self.contract not in slice.bars:
            for chain in slice.futures_chains.values():
                front = sorted(chain.Contracts.values(), key=lambda c: c.Expiry)[0]
                if self.contract != front.Symbol:
                    self.contract = front.Symbol
                    self._ensure(self.contract)
            return

        bar = slice.bars[self.contract]
        ind = self.indicators[self.contract]
        if not (ind["fast"].is_ready and ind["slow"].is_ready and ind["rsi"].is_ready):
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
        rsi_val = ind["rsi"].Current.Value

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

            # LONG only if RSI oversold
            if cross_up and rsi_val <= self.RSI_OS:
                entry = bar.close
                stop  = round(entry - points_risk, 2)
                self.market_order(self.contract,  self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract, -self.CONTRACTS, stop, tag="SL-$100 LONG")

            # SHORT only if RSI over-bought
            elif cross_down and rsi_val >= self.RSI_OB:
                entry = bar.close
                stop  = round(entry + points_risk, 2)
                self.market_order(self.contract, -self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract,  self.CONTRACTS, stop, tag="SL-$100 SHORT")

    # -------------- util helpers ----------------
    def _cancel_stop(self):
        if self.stop_ticket and self.stop_ticket.status in (OrderStatus.NEW,
                                                            OrderStatus.SUBMITTED,
                                                            OrderStatus.PARTIALLY_FILLED):
          self.stop_ticket.cancel()
          
        self.stop_ticket = None

    def on_end_of_algorithm(self):
        self._cancel_stop()