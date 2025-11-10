# region imports
from datetime import time, timedelta
from AlgorithmImports import *
from QuantConnect import Extensions
from QuantConnect.Data.Consolidators import TickConsolidator
from QuantConnect.Indicators import ExponentialMovingAverage, ParabolicStopAndReverse
import pytz
# endregion

class MesGoldenDeathCrossRTH(QCAlgorithm):
    """
    Trades Micro E-mini S&P 500 (MES) on a simple 20 / 25 EMA cross built on 1000-tick bars, but **only** during
    09 : 30 → 16 : 00 US-Eastern.  Any open position is closed no later than 15 : 58 ET.
    SAR-driven trailing stops adapt risk while a trade is open.
    """

    FAST, SLOW = 20, 25
    CONTRACTS  = 1
    STOP_RISK  = 50           # $ stop for the WHOLE position
    POINT_VAL  = 5.0          # $ per index-point for one MES contract
    MIN_STOP_POINTS = 10      # minimum distance from entry for stops
    TICK_SIZE  = 0.25         # MES minimum price increment
    CROSS_BUFFER_TICKS = 2    # ticks beyond crossover to confirm trend

    TZ_NY      = pytz.timezone("America/New_York")
    OPEN_ET    = time(9, 30)
    CLOSE_ET   = time(16, 0)
    FLAT_ET    = time(15, 58)  # hard-flatten deadline
    SYMBOL     = Futures.Indices.SP_500_E_MINI   # CME - "ES"

    # ---------------- initialise -----------------
    def initialize(self):
        self.set_start_date(2025, 11, 5)
        self.set_end_date  (2025, 11, 6)
        self.set_cash(60_000)

        self.universe_settings.resolution = Resolution.TICK

        fut = self.add_future(self.SYMBOL, Resolution.TICK,
                              leverage = 1,
                              extended_market_hours = True,
                              data_normalization_mode=DataNormalizationMode.RAW)
        fut.set_filter(timedelta(0), timedelta(days=90))

        self.contract   = None
        self.indicators = {}          # {Symbol: {'fast','slow','sar'}}
        self.tick_consolidators = {}
        self.latest_bar = {}
        self.last_bar_time = {}
        self.prev_diff  = 0.0
        self.stop_ticket = None
        self.last_roll_date = None
        self.trade_seq = 0
        self.current_tag = ""
        self.current_stop_price = None
        self.awaiting_flat = False

        price_chart = Chart("MES")
        price_chart.add_series(Series("Price", SeriesType.LINE, "", Color.Black))
        price_chart.add_series(Series("Fast EMA", SeriesType.LINE, "", Color.LightGreen))
        price_chart.add_series(Series("Slow EMA", SeriesType.LINE, "", Color.LightCoral))
        price_chart.add_series(Series("SAR", SeriesType.LINE))
        price_chart.add_series(Series("Stop", SeriesType.LINE, "", Color.Blue))
        price_chart.add_series(Series("Long Entry", SeriesType.SCATTER, "", Color.DarkGreen, ScatterMarkerSymbol.TRIANGLE))
        price_chart.add_series(Series("Short Entry", SeriesType.SCATTER, "", Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))
        price_chart.add_series(Series("Long Exit", SeriesType.SCATTER, "", Color.Red, ScatterMarkerSymbol.SQUARE))
        price_chart.add_series(Series("Short Exit", SeriesType.SCATTER, "", Color.DarkGreen, ScatterMarkerSymbol.SQUARE))
        self.add_chart(price_chart)

    # ---------------- indicator helper -----------
    def _ensure(self, symbol: Symbol):
        if symbol in self.indicators:
            return

        consolidator = TickConsolidator(1000)

        def handle_consolidated(sender, bar):
            try:
                self._on_consolidated(symbol, bar)
            except Exception as e:
                Extensions.set_runtime_error(self, e, f"Tick consolidator handler for {symbol.Value}")
                raise

        consolidator.DataConsolidated += handle_consolidated
        self.subscription_manager.add_consolidator(symbol, consolidator)

        fast = ExponentialMovingAverage(f"{symbol.Value}_fast", self.FAST)
        slow = ExponentialMovingAverage(f"{symbol.Value}_slow", self.SLOW)
        sar  = ParabolicStopAndReverse(f"{symbol.Value}_sar")

        def fast_selector(bar: TradeBar):
            try:
                return bar.Close
            except Exception as e:
                Extensions.set_runtime_error(self, e, f"Fast EMA selector for {symbol.Value}")
                raise

        def slow_selector(bar: TradeBar):
            try:
                return bar.Open
            except Exception as e:
                Extensions.set_runtime_error(self, e, f"Slow EMA selector for {symbol.Value}")
                raise

        self.register_indicator(symbol, fast, consolidator, fast_selector)
        self.register_indicator(symbol, slow, consolidator, slow_selector)
        try:
            self.register_indicator(symbol, sar, consolidator)
        except Exception as e:
            Extensions.set_runtime_error(self, e, f"PSAR registration for {symbol.Value}")
            raise

        self.indicators[symbol] = {"fast": fast, "slow": slow, "sar": sar}
        self.tick_consolidators[symbol] = consolidator
        self.last_bar_time[symbol] = None

    # ---------------- on_data --------------------
    def on_data(self, slice: Slice):

        # pick front contract each day
        current_date = self.time.date()
        need_roll = self.contract is None or self.last_roll_date != current_date

        if need_roll:
            if not slice.futures_chains:
                return

            front = None
            for chain in slice.futures_chains.values():
                if not chain.Contracts:
                    continue
                front = sorted(chain.Contracts.values(), key=lambda c: c.Expiry)[0]
                break

            if front is None:
                return

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
            self.latest_bar[self.contract] = None
            self.last_bar_time[self.contract] = None
            self.last_roll_date = current_date
            return

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
        if not (ind["fast"].is_ready and ind["slow"].is_ready and ind["sar"].is_ready):
            return

        # current ET
        t_et = self.time.astimezone(self.TZ_NY).time()

        # force-flat 15:58 or later
        if t_et >= self.FLAT_ET:
            if self.portfolio.invested:
                qty = self.portfolio[self.contract].quantity
                if qty != 0:
                    self._cancel_stop()
                    exit_series = "Long Exit" if qty > 0 else "Short Exit"
                    self.market_order(self.contract, -qty, tag="Forced Flatten EOD")
                    self.plot("MES", exit_series, bar.Close)
                    self.awaiting_flat = True
            return

        # only trade in RTH window
        if not (self.OPEN_ET <= t_et < self.CLOSE_ET):
            return

        # ---------------- EMA cross ----------------
        diff = ind["fast"].Current.Value - ind["slow"].Current.Value
        band = self.CROSS_BUFFER_TICKS * self.TICK_SIZE
        cross_up   = self.prev_diff <= band and diff > band
        cross_down = self.prev_diff >= -band and diff < -band
        self.prev_diff = diff

        sar_val = self._round_price(ind["sar"].Current.Value)
        long_margin = self._round_price(bar.Close - self.MIN_STOP_POINTS)
        short_margin = self._round_price(bar.Close + self.MIN_STOP_POINTS)
        qty = self.portfolio[self.contract].quantity

        if self.awaiting_flat:
            if qty == 0:
                self.awaiting_flat = False
            else:
                return

        self.plot("MES", "Price", bar.Close)
        self.plot("MES", "Fast EMA", ind["fast"].Current.Value)
        self.plot("MES", "Slow EMA", ind["slow"].Current.Value)
        self.plot("MES", "SAR", sar_val)
        if self.current_stop_price is not None:
            self.plot("MES", "Stop", self.current_stop_price)

        # ---------------- exits & stop management --------------------
        if qty > 0:
            # trail long stop up to SAR
            desired_qty = -self.CONTRACTS
            desired_stop = min(sar_val, long_margin)
            if self.stop_ticket is None or self.stop_ticket.quantity != desired_qty:
                self._cancel_stop()
                self.stop_ticket = self.stop_market_order(
                    self.contract, desired_qty, desired_stop,
                    tag=f"STOP LOSS adjusted by SAR Trailing LONG #{self.current_tag or 'NA'}")
                self.current_stop_price = desired_stop
                self.plot("MES", "Stop", desired_stop)
            else:
                current_stop = self.stop_ticket.get(OrderField.STOP_PRICE) or desired_stop
                if desired_stop > current_stop:
                    update = UpdateOrderFields()
                    update.stop_price = desired_stop
                    self.stop_ticket.update(update)
                    self.current_stop_price = desired_stop
                    self.plot("MES", "Stop", desired_stop)

            if cross_down or bar.Close <= sar_val:
                self._cancel_stop()
                exit_series = "Long Exit"
                self.market_order(self.contract, -qty, tag=f"SAR Exit LONG #{self.current_tag or 'NA'}")
                self.plot("MES", exit_series, bar.Close)
                self.awaiting_flat = True
                return

        elif qty < 0:
            desired_qty = self.CONTRACTS
            desired_stop = max(sar_val, short_margin)
            if self.stop_ticket is None or self.stop_ticket.quantity != desired_qty:
                self._cancel_stop()
                self.stop_ticket = self.stop_market_order(
                    self.contract, desired_qty, desired_stop,
                    tag=f"STOP LOSS adjusted by SAR Trailing SHORT #{self.current_tag or 'NA'}")
                self.current_stop_price = desired_stop
                self.plot("MES", "Stop", desired_stop)
            else:
                current_stop = self.stop_ticket.get(OrderField.STOP_PRICE) or desired_stop
                if desired_stop < current_stop:
                    update = UpdateOrderFields()
                    update.stop_price = desired_stop
                    self.stop_ticket.update(update)
                    self.current_stop_price = desired_stop
                    self.plot("MES", "Stop", desired_stop)

            if cross_up or bar.Close >= sar_val:
                self._cancel_stop()
                exit_series = "Short Exit"
                self.market_order(self.contract, -qty, tag=f"SAR Exit SHORT #{self.current_tag or 'NA'}")
                self.plot("MES", exit_series, bar.Close)
                self.awaiting_flat = True
                return

        if qty != 0 or self.awaiting_flat:
            return

        # ---------------- entries -----------------
        points_risk = max(self.STOP_RISK / (self.CONTRACTS * self.POINT_VAL),
                          self.MIN_STOP_POINTS)

        if cross_up:
            self.trade_seq += 1
            self.current_tag = f"{self.trade_seq}"
            entry = bar.Close
            fixed_long_stop = self._round_price(entry - points_risk)
            sar_cap = min(sar_val, self._round_price(entry - self.MIN_STOP_POINTS))
            stop = min(fixed_long_stop, sar_cap)
            self.market_order(self.contract,  self.CONTRACTS, tag=f"ENTRY LONG #{self.current_tag}")
            self.stop_ticket = self.stop_market_order(
                self.contract, -self.CONTRACTS, stop, tag=f"STOP LOSS/SAR LONG #{self.current_tag}")
            self.current_stop_price = stop
            self.plot("MES", "Long Entry", entry)
            self.plot("MES", "Stop", stop)

        elif cross_down:
            self.trade_seq += 1
            self.current_tag = f"{self.trade_seq}"
            entry = bar.Close
            fixed_short_stop = self._round_price(entry + points_risk)
            sar_floor = max(sar_val, self._round_price(entry + self.MIN_STOP_POINTS))
            stop = max(fixed_short_stop, sar_floor)
            self.market_order(self.contract, -self.CONTRACTS, tag=f"ENTRY SHORT #{self.current_tag}")
            self.stop_ticket = self.stop_market_order(
                self.contract,  self.CONTRACTS, stop, tag=f"STOP LOSS/SAR SHORT #{self.current_tag}")
            self.current_stop_price = stop
            self.plot("MES", "Short Entry", entry)
            self.plot("MES", "Stop", stop)

    # -------------- util helpers ----------------
    def _on_consolidated(self, symbol: Symbol, bar: TradeBar):
        self.latest_bar[symbol] = bar

    def _round_price(self, price: float) -> float:
        return round(price / self.TICK_SIZE) * self.TICK_SIZE

    def _cancel_stop(self):
        if self.stop_ticket and self.stop_ticket.status in (OrderStatus.NEW,
                                                            OrderStatus.SUBMITTED,
                                                            OrderStatus.PARTIALLY_FILLED):
            self.stop_ticket.cancel()
        self.stop_ticket = None
        self.current_stop_price = None

    def on_end_of_algorithm(self):
        self._cancel_stop()
        for symbol, consolidator in list(self.tick_consolidators.items()):
            self.subscription_manager.remove_consolidator(symbol, consolidator)
        self.tick_consolidators.clear()
