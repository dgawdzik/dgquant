# region imports
from datetime import time, timedelta
from AlgorithmImports import *
from QuantConnect import Extensions
from QuantConnect.Data.Consolidators import TradeBarConsolidator
from QuantConnect.Indicators import (
    ExponentialMovingAverage,
    ParabolicStopAndReverse,
    AverageDirectionalIndex,
    AverageTrueRange
)
import pytz
from collections import deque
# endregion

class MesGoldenDeathCrossRTH(QCAlgorithm):
    """
    Trades Micro E-mini S&P 500 (MES) on a simple 20 / 50 / 100 EMA fan built on 2-minute bars, but **only** during
    09 : 30 → 16 : 00 US-Eastern.  Any open position is closed no later than 15 : 58 ET.
    SAR-driven trailing stops adapt risk while a trade is open.
    """

    FAST, MID, SLOW = 10, 25, 50
    CONTRACTS  = 1
    STOP_RISK  = 50          # $ stop for the WHOLE position
    POINT_VAL  = 5.0          # $ per index-point for one MES contract
    MIN_STOP_POINTS = 10      # minimum distance from entry for stops
    TICK_SIZE  = 0.25         # MES minimum price increment
    CROSS_BUFFER_TICKS = 2    # ticks beyond crossover to confirm trend
    ADX_PERIOD = 10
    ADX_THRESHOLD = 10
    ATR_PERIOD = 10
    ATR_MIN = 1.2           # minimum ATR in points to avoid chop
    VOLUME_WINDOW = 50
    VOLUME_Z_THRESHOLD = 1.0

    TZ_NY      = pytz.timezone("America/New_York")
    OPEN_ET    = time(9, 30)
    CLOSE_ET   = time(16, 0)
    FLAT_ET    = time(15, 58)  # hard-flatten deadline
    SYMBOL     = Futures.Indices.SP_500_E_MINI   # CME - "ES"

    # ---------------- initialise -----------------
    def initialize(self):
        self.set_start_date(2025, 10, 1)
        self.set_end_date  (2025, 11, 6)
        self.set_cash(60_000)

        self.universe_settings.resolution = Resolution.MINUTE

        fut = self.add_future(self.SYMBOL, Resolution.MINUTE,
                              leverage = 1,
                              extended_market_hours = True,
                              data_normalization_mode=DataNormalizationMode.RAW)
        fut.set_filter(timedelta(0), timedelta(days=90))

        self.contract   = None
        self.indicators = {}          # {Symbol: {'fast','mid','slow','sar'}}
        self.consolidators = {}
        self.volume_history = {}
        self.latest_bar = {}
        self.last_bar_time = {}
        self.prev_diff  = 0.0
        self.stop_ticket = None
        self.last_roll_date = None
        self.trade_seq = 0
        self.current_tag = ""
        self.current_stop_price = None
        self.entry_price = None
        self.awaiting_flat = False
        self.position_side = None

        price_chart = Chart("MES")
        price_chart.add_series(Series("Price", SeriesType.LINE, "", Color.BLACK))
        price_chart.add_series(Series("Fast EMA", SeriesType.LINE, "", Color.LIGHT_GREEN))
        price_chart.add_series(Series("Mid EMA", SeriesType.LINE, "", Color.GOLD))
        price_chart.add_series(Series("Slow EMA", SeriesType.LINE, "", Color.LIGHT_CORAL))
        price_chart.add_series(Series("SAR", SeriesType.LINE))
        price_chart.add_series(Series("Stop", SeriesType.LINE, "", Color.BLUE))
        price_chart.add_series(Series("Long Entry", SeriesType.SCATTER, "", Color.DARK_GREEN, ScatterMarkerSymbol.TRIANGLE))
        price_chart.add_series(Series("Short Entry", SeriesType.SCATTER, "", Color.RED, ScatterMarkerSymbol.TRIANGLE_DOWN))
        price_chart.add_series(Series("Long Exit", SeriesType.SCATTER, "", Color.RED, ScatterMarkerSymbol.SQUARE))
        price_chart.add_series(Series("Short Exit", SeriesType.SCATTER, "", Color.DARK_GREEN, ScatterMarkerSymbol.SQUARE))
        self.add_chart(price_chart)
        self.set_warm_up(timedelta(hours=12))

    # ---------------- indicator helper -----------
    def _ensure(self, symbol: Symbol):
        if symbol in self.indicators:
            return

        consolidator = TradeBarConsolidator(timedelta(minutes=2))

        def handle_consolidated(sender, bar):
            try:
                self._on_consolidated(symbol, bar)
            except Exception as e:
                Extensions.set_runtime_error(self, e, f"Tick consolidator handler for {symbol.value}")
                raise

        consolidator.DataConsolidated += handle_consolidated
        self.subscription_manager.add_consolidator(symbol, consolidator)

        fast = ExponentialMovingAverage(f"{symbol.value}_fast", self.FAST)
        mid  = ExponentialMovingAverage(f"{symbol.value}_mid", self.MID)
        slow = ExponentialMovingAverage(f"{symbol.value}_slow", self.SLOW)
        sar  = ParabolicStopAndReverse(f"{symbol.value}_sar", 0.01, 0.01, 0.1)
        adx  = AverageDirectionalIndex(f"{symbol.value}_adx", self.ADX_PERIOD)
        atr  = AverageTrueRange(f"{symbol.value}_atr", self.ATR_PERIOD, MovingAverageType.WILDERS)

        self.register_indicator(symbol, fast, consolidator, lambda bar: bar.close)
        self.register_indicator(symbol, mid,  consolidator, lambda bar: bar.close)
        self.register_indicator(symbol, slow, consolidator, lambda bar: bar.open)
        try:
            self.register_indicator(symbol, sar, consolidator)
            self.register_indicator(symbol, adx, consolidator)
            self.register_indicator(symbol, atr, consolidator)
        except Exception as e:
            Extensions.set_runtime_error(self, e, f"PSAR registration for {symbol.value}")
            raise

        self.indicators[symbol] = {"fast": fast, "mid": mid, "slow": slow, "sar": sar,
                                   "adx": adx, "atr": atr}
        self.volume_history[symbol] = deque(maxlen=self.VOLUME_WINDOW)
        self.consolidators[symbol] = consolidator
        self.last_bar_time[symbol] = None
        self._warm_up_indicators(symbol, consolidator)

    # ---------------- on_data --------------------
    def on_data(self, slice: Slice):
        if self.is_warming_up:
            return

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
                    consolidator = self.consolidators.pop(self.contract, None)
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
        if not all(ind[name].is_ready for name in ("fast", "mid", "slow", "sar", "adx", "atr")):
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
                    self.current_tag = ""
                    self.position_side = None
                    self.awaiting_flat = True
            return

        # only trade in RTH window
        if not (self.OPEN_ET <= t_et < self.CLOSE_ET):
            return

        # ---------------- EMA cross ----------------
        fast_val = ind["fast"].Current.Value
        mid_val  = ind["mid"].Current.Value
        slow_val = ind["slow"].Current.Value

        diff = fast_val - slow_val
        cross_up   = self.prev_diff <= 0 < diff
        cross_down = self.prev_diff >= 0 > diff
        self.prev_diff = diff
        bull_fan = fast_val > mid_val > slow_val
        bear_fan = fast_val < mid_val < slow_val

        sar_val = self._round_price(ind["sar"].Current.Value)
        adx_val = ind["adx"].Current.Value
        atr_val = ind["atr"].Current.Value
        trend_strength = adx_val >= self.ADX_THRESHOLD
        atr_ok = atr_val >= self.ATR_MIN
        long_margin = self._round_price(bar.Close - self.MIN_STOP_POINTS)
        short_margin = self._round_price(bar.Close + self.MIN_STOP_POINTS)
        qty = self.portfolio[self.contract].quantity
        vol_hist = self.volume_history.setdefault(self.contract, deque(maxlen=self.VOLUME_WINDOW))
        vol_hist.append(bar.Volume)
        if len(vol_hist) < self.VOLUME_WINDOW:
            return
        vol_mean = sum(vol_hist) / len(vol_hist)
        vol_std = (sum((v - vol_mean) ** 2 for v in vol_hist) / len(vol_hist)) ** 0.5
        vol_z = (bar.Volume - vol_mean) / vol_std if vol_std > 0 else 0

        if self.awaiting_flat:
            if qty == 0:
                self.awaiting_flat = False
            else:
                return
        self.plot("MES", "Price", bar.Close)
        self.plot("MES", "Fast EMA", fast_val)
        self.plot("MES", "Mid EMA", mid_val)
        self.plot("MES", "Slow EMA", slow_val)
        self.plot("MES", "SAR", sar_val)
        if self.current_stop_price is not None:
            self.plot("MES", "Stop", self.current_stop_price)

        # ---------------- exits & stop management --------------------
        if qty > 0:
            sar_stop = self._round_price(min(sar_val, slow_val))
            loss_floor = self._round_price(self.entry_price - self.STOP_RISK / self.POINT_VAL)
            stop_level = max(loss_floor, sar_stop)
            if self.stop_ticket is None or self.stop_ticket.quantity != -self.CONTRACTS:
                self._cancel_stop()
                self.stop_ticket = self.stop_market_order(
                    self.contract, -self.CONTRACTS, stop_level,
                    tag=f"STOP LOSS/SAR LONG #{self.current_tag or 'NA'}")
            else:
                current_stop = self.stop_ticket.get(OrderField.STOP_PRICE) or stop_level
                if stop_level > current_stop:
                    update = UpdateOrderFields()
                    update.stop_price = stop_level
                    self.stop_ticket.update(update)
            self.current_stop_price = stop_level
            self.plot("MES", "Stop", stop_level)

        elif qty < 0:
            sar_stop = self._round_price(max(sar_val, slow_val))
            loss_ceiling = self._round_price(self.entry_price + self.STOP_RISK / self.POINT_VAL)
            stop_level = min(loss_ceiling, sar_stop)
            if self.stop_ticket is None or self.stop_ticket.quantity != self.CONTRACTS:
                self._cancel_stop()
                self.stop_ticket = self.stop_market_order(
                    self.contract, self.CONTRACTS, stop_level,
                    tag=f"STOP LOSS/SAR SHORT #{self.current_tag or 'NA'}")
            else:
                current_stop = self.stop_ticket.get(OrderField.STOP_PRICE) or stop_level
                if stop_level < current_stop:
                    update = UpdateOrderFields()
                    update.stop_price = stop_level
                    self.stop_ticket.update(update)
            self.current_stop_price = stop_level
            self.plot("MES", "Stop", stop_level)

        if qty != 0:
            return

        # ---------------- entries -----------------
        points_risk = max(self.STOP_RISK / (self.CONTRACTS * self.POINT_VAL),
                          self.MIN_STOP_POINTS)

        if cross_up and bull_fan and trend_strength and atr_ok and vol_z >= self.VOLUME_Z_THRESHOLD:
            self.trade_seq += 1
            self.current_tag = f"{self.trade_seq}"
            entry = bar.Close
            sar_cap = min(sar_val, slow_val)
            loss_floor = self._round_price(entry - self.STOP_RISK / self.POINT_VAL)
            stop = max(loss_floor, sar_cap)
            self.market_order(self.contract,  self.CONTRACTS, tag=f"ENTRY LONG #{self.current_tag}")
            self.stop_ticket = self.stop_market_order(
                self.contract, -self.CONTRACTS, stop, tag=f"STOP LOSS/SAR LONG #{self.current_tag}")
            self.current_stop_price = stop
            self.position_side = "long"
            self.entry_price = entry
            self.plot("MES", "Long Entry", entry)
            self.plot("MES", "Stop", stop)

        elif cross_down and bear_fan and trend_strength and atr_ok and vol_z >= self.VOLUME_Z_THRESHOLD:
            self.trade_seq += 1
            self.current_tag = f"{self.trade_seq}"
            entry = bar.Close
            sar_floor = max(sar_val, slow_val)
            loss_ceiling = self._round_price(entry + self.STOP_RISK / self.POINT_VAL)
            stop = min(loss_ceiling, sar_floor)
            self.market_order(self.contract, -self.CONTRACTS, tag=f"ENTRY SHORT #{self.current_tag}")
            self.stop_ticket = self.stop_market_order(
                self.contract,  self.CONTRACTS, stop, tag=f"STOP LOSS/SAR SHORT #{self.current_tag}")
            self.current_stop_price = stop
            self.position_side = "short"
            self.entry_price = entry
            self.plot("MES", "Short Entry", entry)
            self.plot("MES", "Stop", stop)

    # -------------- util helpers ----------------
    def _on_consolidated(self, symbol: Symbol, bar: TradeBar):
        self.latest_bar[symbol] = bar

    def _warm_up_indicators(self, symbol: Symbol, consolidator: TradeBarConsolidator):
        if self.is_warming_up:
            return

        history = self.history(symbol, 500, Resolution.MINUTE)
        if symbol not in history.index.get_level_values(0):
            return
        for time, row in history.loc[symbol].iterrows():
            consolidator.update(TradeBar(
                time=time,
                symbol=symbol,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                period=timedelta(minutes=1)))

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
        for symbol, consolidator in list(self.consolidators.items()):
            self.subscription_manager.remove_consolidator(symbol, consolidator)
        self.consolidators.clear()

    def on_order_event(self, order_event: OrderEvent):
        if not self.stop_ticket or order_event.order_id != self.stop_ticket.order_id:
            return
        if order_event.status != OrderStatus.FILLED:
            return

        exit_series = "Long Exit" if order_event.direction == OrderDirection.SELL else "Short Exit"
        self.plot("MES", exit_series, order_event.fill_price)
        self.current_stop_price = None
        self.position_side = None
        self.current_tag = ""
        self.awaiting_flat = True
        self.entry_price = None
        self.stop_ticket = None
