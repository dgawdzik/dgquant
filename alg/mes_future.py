# region imports
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

    # --------------- initialise ----------------
    def initialize(self):
        self.set_start_date(2025, 4, 25)
        self.set_end_date  (2025, 4, 25)
        self.set_cash(40_000)

        self.universe_settings.resolution = Resolution.MINUTE

        fut  = self.add_future(self.SYMBOL, Resolution.MINUTE,
                               data_normalization_mode=DataNormalizationMode.Raw)
        fut.set_filter(timedelta(0), timedelta(days=90))

        self.contract = None
        self.indicators = {}
        self.prev_diff = 0.0
        self.stop_ticket = None

    # --------------- indicator helper ----------
    def _ensure(self, symbol):
        if symbol in self.indicators:
            return
        fast = self.ema(symbol, self.FAST, Resolution.MINUTE)
        slow = self.ema(symbol, self.SLOW, Resolution.MINUTE)
        self.indicators[symbol] = {"fast": fast, "slow": slow}

    # ---------------- on_data -------------------
    def on_data(self, slice: Slice):

        # --- select front contract daily ---
        if self.contract is None or self.contract not in slice.Bars:
            for chain in slice.FutureChains.values():
                front = sorted(chain.Contracts.values(), key=lambda c: c.Expiry)[0]
                if self.contract != front.Symbol:
                    self.contract = front.Symbol
                    self._ensure(self.contract)
            return

        bar = slice.Bars[self.contract]
        ind = self.indicators[self.contract]
        if not (ind["fast"].IsReady and ind["slow"].IsReady):
            return

        # --- current time in Eastern ---
        t_et = self.Time.astimezone(self.TZ_NY).time()

        # hard-flatten at 15:58 ET
        if t_et >= self.FLAT_ET:
            if self.Portfolio.Invested:
                self.liquidate(self.contract)
                self._cancel_stop()
            return                                      # no new trades

        # trade only inside 09:30–16:00 window
        if not (self.OPEN_ET <= t_et < self.CLOSE_ET):
            return

        # ------- EMA cross logic ----------
        diff  = ind["fast"].Current.Value - ind["slow"].Current.Value
        cross_up   = self.prev_diff <= 0 < diff
        cross_down = self.prev_diff >= 0 > diff
        self.prev_diff = diff

        qty = self.Portfolio[self.contract].Quantity

        # ---------- exits ----------
        if qty > 0 and cross_down:       # long -> exit
            self._cancel_stop()
            self.liquidate(self.contract)
            qty = 0
        elif qty < 0 and cross_up:       # short -> exit
            self._cancel_stop()
            self.liquidate(self.contract)
            qty = 0

        # ---------- entries ----------
        if qty == 0:
            points_risk = self.STOP_RISK / (self.CONTRACTS * self.POINT_VAL)
            if cross_up:                             # go LONG
                entry = bar.Close
                stop  = round(entry - points_risk, 2)
                self.market_order(self.contract,  self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract, -self.CONTRACTS, stop, tag="SL-$100 LONG")
            elif cross_down:                         # go SHORT
                entry = bar.Close
                stop  = round(entry + points_risk, 2)
                self.market_order(self.contract, -self.CONTRACTS)
                self.stop_ticket = self.stop_market_order(
                    self.contract,  self.CONTRACTS, stop, tag="SL-$100 SHORT")

    # -------------- util helpers ---------------
    def _cancel_stop(self):
        if self.stop_ticket and self.stop_ticket.status in (OrderStatus.NEW,
                                                            OrderStatus.SUBMITTED,
                                                            OrderStatus.PARTIALLY_FILLED):
            self.stop_ticket.cancel()
        self.stop_ticket = None

    def on_end_of_algorithm(self):
        self._cancel_stop()