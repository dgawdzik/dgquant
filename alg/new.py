# region imports
from AlgorithmImports import *
import statistics
from collections import deque
# endregion


class OneMinEMACrossMacdBias(QCAlgorithm):
    """
    1-minute 8/21 EMA-crossover strategy that trades **only when
    BOTH the 1-minute and the 5-minute MACD agree** on direction.

      • enter on EMA cross *and* MACD agreement
      • exit when the fast EMA breaks back through the slow EMA
        or the 1.2 × ATR stop is hit
      • volume & ATR filters
      • risk-based sizing plus buying-power buffer
    Compatible with LEAN v17076 (no 5-min Resolution helper, uses a consolidator).
    """

    FAST, SLOW = 23, 50
    RISK_PCT   = 0.02
    SYMBOL     = "NVDA"
    BP_BUFFER  = 100

    # MACD parameters
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIG  = 9

    SESSION_START = time(9, 35)
    SESSION_END   = time(15, 55)

    # --------------- INITIALISE -----------------
    def initialize(self):
        self.set_start_date(2025, 1, 13)
        self.set_end_date  (2025, 1, 17)
        self.set_cash(10_000)

        self.universe_settings.resolution            = Resolution.MINUTE
        self.universe_settings.extended_market_hours = True

        self.sym = self.add_equity(
            self.SYMBOL,
            Resolution.MINUTE,
            leverage              = 4,
            fill_forward          = True,
            extended_market_hours = True,
            data_normalization_mode = DataNormalizationMode.Raw
        ).symbol

        # EMA crossover indicators
        self.fast_ema = self.ema(self.sym, self.FAST, Resolution.MINUTE)
        self.slow_ema = self.ema(self.sym, self.SLOW, Resolution.MINUTE)

        # MACD on 1-minute bars
        self.macd_1m = self.macd(
            self.sym,
            self.MACD_FAST,
            self.MACD_SLOW,
            self.MACD_SIG,
            MovingAverageType.Exponential,
            Resolution.MINUTE
        )

        # MACD on 5-minute bars via consolidator
        self.macd_5m = MovingAverageConvergenceDivergence(
            self.MACD_FAST, self.MACD_SLOW, self.MACD_SIG, MovingAverageType.Exponential
        )
        five_min = TradeBarConsolidator(timedelta(minutes=5))
        five_min.DataConsolidated += lambda s, bar: self.macd_5m.Update(bar.EndTime, bar.Close)
        self.SubscriptionManager.AddConsolidator(self.sym, five_min)

        # Filters
        self.atr       = self.atr(self.sym, 14, MovingAverageType.Simple, Resolution.MINUTE)
        self.vol_sma20 = self.sma(self.sym, 20, Resolution.MINUTE, Field.Volume)
        self.atr_vals: deque[float] = deque(maxlen=20)

        self.set_warm_up(self.SLOW)
        self.prev_diff   = 0.0

        # position state
        self.position_dir = 0      # +1 long | -1 short | 0 flat
        self.stop_price   = None

    # --------------- HELPERS -------------------
    def in_session(self):
        return self.SESSION_START <= self.Time.time() <= self.SESSION_END

    # ---------- MACD-direction helpers ----------
    def macd_bullish(self) -> bool:
        return (
            self.macd_1m.IsReady
            and self.macd_5m.IsReady
            and self.macd_1m.Current.Value > self.macd_1m.Signal.Current.Value
            and self.macd_5m.Current.Value > self.macd_5m.Signal.Current.Value
        )

    def macd_bearish(self) -> bool:
        return (
            self.macd_1m.IsReady
            and self.macd_5m.IsReady
            and self.macd_1m.Current.Value < self.macd_1m.Signal.Current.Value
            and self.macd_5m.Current.Value < self.macd_5m.Signal.Current.Value
        )

    def position_size(self, stop_dollar: float, price: float) -> int:
        risk_cap = self.Portfolio.TotalPortfolioValue * self.RISK_PCT
        qty_risk = risk_cap / stop_dollar

        bp_left  = max(0, self.Portfolio.MarginRemaining - self.BP_BUFFER)
        qty_bp   = bp_left / price

        lot = self.Securities[self.sym].SymbolProperties.LotSize
        qty = int(min(qty_risk, qty_bp) // lot * lot)
        return max(lot if qty == 0 else qty, 0)

    # --------------- MAIN LOOP -----------------
    def on_data(self, data: Slice):

        if self.is_warming_up or self.sym not in data.bars:
            return

        bar = data.bars[self.sym]

        # ---------- exit management -------------
        fast_val = self.fast_ema.Current.Value
        slow_val = self.slow_ema.Current.Value
        if self.position_dir == 1:
            if bar.Low <= self.stop_price or fast_val < slow_val:
                self.liquidate(self.sym)
                self.position_dir = 0
        elif self.position_dir == -1:
            if bar.High >= self.stop_price or fast_val > slow_val:
                self.liquidate(self.sym)
                self.position_dir = 0

        # ---------- session guard --------------
        if not self.in_session():
            if self.Portfolio.Invested:
                self.liquidate(self.sym)
                self.position_dir = 0
            return

        # ---------- filters --------------------
        if not self.vol_sma20.IsReady or not self.atr.IsReady or not self.macd_5m.IsReady:
            return

        vol_ok = bar.Volume >= 0.5 * self.vol_sma20.Current.Value

        atr_val = self.atr.Current.Value
        self.atr_vals.append(atr_val)
        volat_ok = len(self.atr_vals) == 20 and atr_val > statistics.median(self.atr_vals)

        if not (vol_ok and volat_ok):
            return

        # ---------- crossover trigger ----------
        diff       = fast_val - slow_val
        cross_up   = self.prev_diff <= 0 < diff
        cross_down = self.prev_diff >= 0 > diff
        self.prev_diff = diff

        stop_dist = atr_val * 1.2
        qty       = self.position_size(stop_dist, bar.Close)
        if qty == 0:
            return

        # ---------- entries --------------------
        if self.position_dir == 0:                           # only enter if flat
            if cross_up and self.macd_bullish():
                self.buy(self.sym, qty)
                self.position_dir = 1
                self.stop_price   = bar.Close - stop_dist

            elif cross_down and self.macd_bearish():
                self.sell(self.sym, qty)                    # open short
                self.position_dir = -1
                self.stop_price   = bar.Close + stop_dist