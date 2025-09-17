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
    SYMBOL     = "HOOD"
    BP_BUFFER  = 100

    # MACD parameters
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIG  = 9

    SESSION_START = time(4, 0)
    SESSION_END   = time(15, 55)

    # --------------- INITIALISE -----------------
    def initialize(self):
        self.set_start_date(2025, 8, 1)
        self.set_end_date  (2025, 9, 15)
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
        five_min.data_consolidated += lambda s, bar: self.macd_5m.update(bar.end_time, bar.close)
        self.subscription_manager.add_consolidator(self.sym, five_min)

        # Parabolic SAR for trailing-stop management
        self.psar_indicator = self.psar(self.sym, 0.02, 0.02, 0.2)

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
        return self.SESSION_START <= self.time.time() <= self.SESSION_END

    def has_open_orders(self) -> bool:
        return any(self.transactions.get_open_orders(self.sym))

    # ---------- MACD-direction helpers ----------
    def macd_bullish(self) -> bool:
        return (
            self.macd_1m.is_ready
            and self.macd_5m.is_ready
            and self.macd_1m.current.value > self.macd_1m.signal.current.value
            and self.macd_5m.current.value > self.macd_5m.signal.current.value
        )

    def macd_bearish(self) -> bool:
        return (
            self.macd_1m.is_ready
            and self.macd_5m.is_ready
            and self.macd_1m.current.value < self.macd_1m.signal.current.value
            and self.macd_5m.current.value < self.macd_5m.signal.current.value
        )

    def position_size(self, stop_dollar: float, price: float) -> int:
        risk_cap = self.portfolio.total_portfolio_value * self.RISK_PCT
        qty_risk = risk_cap / stop_dollar

        bp_left  = max(0, self.portfolio.margin_remaining - self.BP_BUFFER)
        qty_bp   = bp_left / price

        lot = self.securities[self.sym].symbol_properties.lot_size
        qty = int(min(qty_risk, qty_bp) // lot * lot)
        return max(lot if qty == 0 else qty, 0)

    # --------------- MAIN LOOP -----------------
    def on_data(self, data: Slice):

        if self.is_warming_up or self.sym not in data.bars:
            return

        bar = data.bars[self.sym]

        psar_value = self.psar_indicator.current.value if self.psar_indicator.is_ready else None

        # ---------- exit management -------------

        if self.position_dir == 1:
            if psar_value is not None:
                self.stop_price = psar_value if self.stop_price is None else max(self.stop_price, psar_value)

            if self.stop_price is not None and (bar.low <= self.stop_price or self.macd_bearish() or not self.in_session()):
                qty_to_exit = -self.portfolio[self.sym].quantity
                if qty_to_exit > 0:
                    self.limit_order(self.sym, qty_to_exit, bar.close - 0.2)
                self.position_dir = 0
                self.stop_price   = None
        elif self.position_dir == -1:
            if psar_value is not None:
                self.stop_price = psar_value if self.stop_price is None else min(self.stop_price, psar_value)

            if self.stop_price is not None and (bar.high >= self.stop_price or self.macd_bullish() or not self.in_session()):
                qty_to_cover = -self.portfolio[self.sym].quantity
                if qty_to_cover > 0:
                    self.limit_order(self.sym, qty_to_cover, bar.close + 0.2)
                self.position_dir = 0
                self.stop_price   = None

        # ---------- session guard --------------
        if not self.in_session():
            if self.portfolio.invested or self.has_open_orders():
                self.liquidate(self.sym)
                self.position_dir = 0
                self.stop_price   = None
            return

        # ---------- filters --------------------
        if not self.vol_sma20.is_ready or not self.atr.is_ready or not self.macd_5m.is_ready or not self.psar_indicator.is_ready:
            return

        vol_ok = bar.volume >= 1.5 * self.vol_sma20.current.value

        atr_val = self.atr.current.value
        self.atr_vals.append(atr_val)
        volat_ok = len(self.atr_vals) == 20 and atr_val > statistics.median(self.atr_vals) * 1.5

        if not (vol_ok and volat_ok):
            return

        # ---------- crossover trigger ----------
        fast_val = self.fast_ema.current.value
        slow_val = self.slow_ema.current.value
        diff       = fast_val - slow_val
        cross_up   = self.prev_diff <= 0 < diff
        cross_down = self.prev_diff >= 0 > diff
        self.prev_diff = diff

        # ---------- entries --------------------
        if self.position_dir == 0 and not self.portfolio.invested and not self.has_open_orders():               # only enter if flat
            if cross_up and self.macd_bullish() and psar_value is not None and psar_value < bar.close:
                stop_dist = bar.close - psar_value
                qty       = self.position_size(stop_dist, bar.close)
                if qty > 0:
                    self.limit_order(self.sym, qty, bar.close)
                    self.position_dir = 1
                    self.stop_price   = psar_value

            elif cross_down and self.macd_bearish() and psar_value is not None and psar_value > bar.close:
                stop_dist = psar_value - bar.close
                qty       = self.position_size(stop_dist, bar.close)
                if qty > 0:
                    self.limit_order(self.sym, -qty, bar.close)                    # open short
                    self.position_dir = -1
                    self.stop_price   = psar_value
