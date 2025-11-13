# region imports
from AlgorithmImports import *
from datetime import datetime, timedelta, time
from collections import deque
from typing import Tuple
import numpy as np
import pandas as pd
import pytz
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
# endregion


class MesXGBoost(QCAlgorithm):
    """
    Event-driven MES futures strategy that uses an XGBoost classifier to predict
    short-horizon direction from engineered price-structure features.
    """

    LABEL_HORIZON_MIN = 5        # forward minutes to build the classification label
    MIN_TRAIN_SAMPLES = 200      # minimum samples needed before the first fit
    MAX_TRAIN_SAMPLES = 3000     # rolling cap to avoid unbounded memory
    PROB_LONG_ENTER = 0.80       # probability threshold to enter long
    PROB_LONG_EXIT  = 0.60       # probability threshold to stay long
    PROB_SHORT_ENTER = 0.20      # probability threshold to enter short
    PROB_SHORT_EXIT  = 0.40      # probability threshold to stay short
    RETURN_THRESHOLD = 0.0       # label = 1 if future return > threshold else 0
    MIN_TRAIN_DAYS = 25          # days of data to accumulate before trading
    CONTRACTS = 1                # number of MES contracts to trade
    PREV_30M_DAYS = 10
    MAX_30M_SLOTS = 20
    ROLL_DAYS_BEFORE_EXPIRY = 8  # switch to next contract this many days before expiry

    FLAT_ET = time(15, 58)       # force flat no later than 15:58 ET
    TRADE_EXTENDED_HOURS = False  # trade outside NY RTH when True
    REGULAR_MARKET_OPEN_ET = time(9, 30)
    REGULAR_MARKET_CLOSE_ET = FLAT_ET
    GLOBAL_CLOSE_START_ET = time(17, 0)  # CME daily maintenance window (all markets closed)
    GLOBAL_CLOSE_END_ET = time(18, 0)

    def initialize(self) -> None:
        self.set_start_date(2025, 10, 1)
        self.set_end_date(2025, 11, 10)
        self.set_cash(75_000)

        self.universe_settings.resolution = Resolution.MINUTE
        future = self.add_future(
            Futures.Indices.SP_500_E_MINI,
            Resolution.MINUTE,
            leverage=1,
            data_normalization_mode=DataNormalizationMode.RAW,
            extended_market_hours=True
        )
        future.set_filter(timedelta(0), timedelta(days=90))
        self.future_chain_symbol = future.symbol

        self.contract_symbol: Symbol | None = None
        self.last_roll_date = None
        self.consolidators: list[Tuple[Symbol, IDataConsolidator]] = []

        self.price_history = deque(maxlen=60)       # 1-minute closes
        self.return_history = deque(maxlen=120)     # 1-minute returns for vol/skew/kurt
        self.volume_history = deque(maxlen=60)      # 1-minute volumes for normalization
        self.pending_labels = deque()
        self.training_features: list[list[float]] = []
        self.training_labels: list[int] = []

        self.vwap_volume = 0.0
        self.vwap_pv = 0.0
        self.session_date = None

        self.prev_day_high = None
        self.prev_day_low = None

        self.prev_day_30m = deque(maxlen=self.PREV_30M_DAYS)
        self.current_day_30m: list[Tuple[float, float, float, float, float]] = []
        self.last_5m_bar: TradeBar | None = None

        # technical indicators
        self.ema_indicator = None
        self.macd_indicator = None
        self.rsi_indicator = None
        self.atr_indicator = None
        self.sar_indicator = None
        self.prev_ema_val = None
        self.prev_macd_hist = None
        self.prev_rsi_val = None

        self.model = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            objective="binary:logistic",
            reg_lambda=1.0,
            verbosity=0
        ) if XGBClassifier else None
        self.model_trained = False
        self.pretraining_done = False

        self.set_warm_up(timedelta(days=2))
        self.training_unlock_time = self.StartDate + timedelta(days=self.MIN_TRAIN_DAYS)

    # ---------------- data loop -----------------
    def on_data(self, slice: Slice) -> None:
        self._ensure_contract(slice)

    # --------------- helpers --------------------
    def _ensure_contract(self, slice: Slice) -> None:
        """Select front MES contract once per day and wire consolidators."""
        current_date = self.time.date()
        need_roll = self.contract_symbol is None or self.last_roll_date != current_date
        if not need_roll:
            return

        if not slice.future_chains:
            return

        front = None
        roll_cutoff = current_date + timedelta(days=self.ROLL_DAYS_BEFORE_EXPIRY)
        for chain in slice.future_chains.values():
            if not chain.Contracts:
                continue
            sorted_contracts = sorted(chain.Contracts.values(), key=lambda c: c.Expiry)
            front = next(
                (contract for contract in sorted_contracts if contract.Expiry.date() > roll_cutoff),
                sorted_contracts[0]
            )
            break

        if front is None:
            return

        switched = self.contract_symbol != front.Symbol
        if self.contract_symbol and switched:
            self._remove_consolidators()

        self.contract_symbol = front.Symbol
        self.last_roll_date = current_date
        self._setup_consolidators(front.Symbol)
        if switched or self.ema_indicator is None:
            self._init_indicators(front.Symbol)
        if not self.pretraining_done and self.contract_symbol:
            self._run_pretraining(self.contract_symbol)

    def _setup_consolidators(self, symbol: Symbol) -> None:
        self._remove_consolidators()

        con5 = TradeBarConsolidator(timedelta(minutes=5))

        def handle_5m(sender, bar):
            self._on_consolidated("5m", bar)
            self._process_primary_bar(bar)

        con5.DataConsolidated += handle_5m
        self.subscription_manager.add_consolidator(symbol, con5)
        self.consolidators.append((symbol, con5))

        con30 = TradeBarConsolidator(timedelta(minutes=30))
        con30.DataConsolidated += lambda sender, bar: self._record_30m_bar(bar)
        self.subscription_manager.add_consolidator(symbol, con30)
        self.consolidators.append((symbol, con30))

        daily = self.consolidate(symbol, Resolution.Daily,
                                 lambda bar: self._on_daily(symbol, bar))
        self.consolidators.append((symbol, daily))

    def _init_indicators(self, symbol: Symbol) -> None:
        self.ema_indicator = self.EMA(symbol, 20, Resolution.MINUTE)
        self.macd_indicator = self.MACD(symbol, 12, 26, 9, MovingAverageType.EXPONENTIAL, Resolution.MINUTE)
        self.rsi_indicator = self.RSI(symbol, 14, MovingAverageType.WILDERS, Resolution.MINUTE)
        self.atr_indicator = self.ATR(symbol, 14, MovingAverageType.WILDERS, Resolution.MINUTE)
        self.sar_indicator = self.PSAR(symbol, 0.02, 0.02, 0.2, Resolution.MINUTE)
        self.prev_ema_val = None
        self.prev_macd_hist = None
        self.prev_rsi_val = None

    def _run_pretraining(self, symbol: Symbol) -> None:
        if self.pretraining_done or self.model is None:
            return
        
        df = self._build_rolling_history_dataframe()
        if df is None or df.empty:
            self.Log("Pretraining skipped: unable to assemble rolling history.")
            self.pretraining_done = True
            return
        temp5 = TradeBarConsolidator(timedelta(minutes=5))
        temp5.DataConsolidated += lambda sender, bar: self._process_primary_bar(bar, is_pretrain=True)
        temp30 = TradeBarConsolidator(timedelta(minutes=30))
        temp30.DataConsolidated += lambda sender, bar: self._record_30m_bar(bar, is_pretrain=True)
        for time_index, row in df.iterrows():
            bar_time = self._coerce_datetime(time_index)
            if bar_time is None:
                continue
            bar_symbol = self._extract_symbol_from_index(time_index, symbol)
            open_val = float(row.open)
            high_val = float(row.high)
            low_val = float(row.low)
            close_val = float(row.close)
            volume_val = float(row.volume)
            bar = TradeBar(
                time=bar_time,
                symbol=bar_symbol,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close_val,
                volume=volume_val,
                period=timedelta(minutes=1)
            )
            temp5.Update(bar)
            temp30.Update(bar)
        temp5.Dispose()
        temp30.Dispose()
        self.pending_labels.clear()
        prev_unlock = self.training_unlock_time
        self._maybe_train_model(force=True)
        if self.model_trained:
            self.training_unlock_time = min(self.training_unlock_time, self.StartDate)
        else:
            self.training_unlock_time = prev_unlock
        self.pretraining_done = True
        self.Log("Pretraining finished; live/paper training now active.")

    def _remove_consolidators(self) -> None:
        for symbol, consolidator in self.consolidators:
            self.subscription_manager.remove_consolidator(symbol, consolidator)
        self.consolidators.clear()

    def _on_consolidated(self, tf: str, bar: TradeBar) -> None:
        if tf == "5m":
            self.last_5m_bar = bar

    def _record_30m_bar(self, bar: TradeBar, is_pretrain: bool = False) -> None:
        vwap = (bar.Open + bar.High + bar.Low + bar.Close) / 4 if bar.Volume == 0 else bar.Close
        time_feats = self._time_features(bar.EndTime)
        slot = (bar.Open, bar.Close, bar.High, bar.Low, vwap, time_feats[0], time_feats[1])
        self.current_day_30m.append(slot)
        if len(self.current_day_30m) > self.MAX_30M_SLOTS:
            self.current_day_30m.pop(0)
    def _on_daily(self, symbol: Symbol, bar: TradeBar) -> None:
        # use most recent completed session for next-day normalization
        self.prev_day_high = bar.High
        self.prev_day_low = bar.Low

    def _reset_session_if_needed(self, bar_time: datetime) -> None:
        date = bar_time.astimezone(pytz.timezone("America/New_York")).date()
        if self.session_date is None:
            self.session_date = date
            self.vwap_volume = 0.0
            self.vwap_pv = 0.0
            return
        if self.session_date == date:
            return
        if self.current_day_30m:
            self.prev_day_30m.append(list(self.current_day_30m))
            self.current_day_30m = []
        self.session_date = date
        self.vwap_volume = 0.0
        self.vwap_pv = 0.0

    def _coerce_datetime(self, index_value) -> datetime | None:
        candidates = list(index_value) if isinstance(index_value, tuple) else [index_value]
        # prioritize last elements (e.g., (symbol, timestamp))
        for candidate in reversed(candidates):
            try:
                if isinstance(candidate, datetime):
                    return candidate
                if isinstance(candidate, pd.Timestamp):
                    return candidate.to_pydatetime()
                if isinstance(candidate, np.datetime64):
                    return pd.Timestamp(candidate).to_pydatetime()
                if hasattr(candidate, "to_pydatetime"):
                    return candidate.to_pydatetime()
                return pd.to_datetime(candidate).to_pydatetime()
            except Exception:
                continue
        self.Log(f"Pretraining skipped row: unable to coerce datetime from index value {index_value}")
        return None

    def _extract_symbol_from_index(self, index_value, fallback: Symbol | None = None) -> Symbol | None:
        if isinstance(index_value, tuple) and index_value:
            candidate = index_value[0]
            if isinstance(candidate, Symbol):
                return candidate
        return fallback

    def _update_vwap(self, bar: TradeBar) -> None:
        self.vwap_volume += bar.Volume
        self.vwap_pv += bar.Close * bar.Volume

    def _is_time_in_window(self, current: time, start: time, end: time) -> bool:
        if start <= end:
            return start <= current < end
        return current >= start or current < end

    def _is_regular_session(self, current: time) -> bool:
        return self._is_time_in_window(current, self.REGULAR_MARKET_OPEN_ET, self.REGULAR_MARKET_CLOSE_ET)

    def _is_global_market_closed(self, current: time) -> bool:
        return self._is_time_in_window(current, self.GLOBAL_CLOSE_START_ET, self.GLOBAL_CLOSE_END_ET)

    def _contract_symbol_for_date(self, target_date: date) -> Symbol | None:
        if not self.future_chain_symbol:
            return None
        request_time = datetime.combine(target_date, time.min)
        contracts = self.FutureChainProvider.GetFutureContractList(self.future_chain_symbol, request_time)
        if not contracts:
            return None
        sorted_contracts = sorted(contracts, key=lambda s: s.ID.Date)
        roll_cutoff = target_date + timedelta(days=self.ROLL_DAYS_BEFORE_EXPIRY)
        for candidate in sorted_contracts:
            expiry = candidate.ID.Date.date()
            if expiry > roll_cutoff:
                return candidate
        return sorted_contracts[0]

    def _build_rolling_history_dataframe(self) -> pd.DataFrame | None:
        end_time = self.time
        if end_time == datetime.min:
            end_time = self.StartDate + timedelta(days=self.MIN_TRAIN_DAYS)
        end_date = end_time.date()
        start_date = end_date - timedelta(days=self.MIN_TRAIN_DAYS)
        frames: list[pd.DataFrame] = []
        days_collected = 0
        current_date = start_date
        max_iterations = self.MIN_TRAIN_DAYS + 10
        while current_date <= end_date and max_iterations > 0 and days_collected < self.MIN_TRAIN_DAYS:
            contract_symbol = self._contract_symbol_for_date(current_date)
            day_start = datetime.combine(current_date, time.min)
            day_end = min(day_start + timedelta(days=1), end_time)
            if not contract_symbol or day_end <= day_start:
                current_date += timedelta(days=1)
                max_iterations -= 1
                continue
            history = self.History(contract_symbol, day_start, day_end, Resolution.MINUTE)
            if history.empty:
                current_date += timedelta(days=1)
                max_iterations -= 1
                continue
            frames.append(history)
            days_collected += 1
            current_date += timedelta(days=1)
            max_iterations -= 1
        if days_collected < self.MIN_TRAIN_DAYS or not frames:
            return None
        df = pd.concat(frames).sort_index()
        return df

    def _build_features(self, bar: TradeBar) -> list[float] | None:
        prev_close = self.price_history[-1] if self.price_history else None
        self.price_history.append(bar.Close)
        self.volume_history.append(bar.Volume)

        if prev_close is None or prev_close <= 0:
            return None

        if not all([
            self.ema_indicator and self.ema_indicator.IsReady,
            self.macd_indicator and self.macd_indicator.IsReady,
            self.rsi_indicator and self.rsi_indicator.IsReady,
            self.atr_indicator and self.atr_indicator.IsReady,
            self.sar_indicator and self.sar_indicator.IsReady
        ]):
            return None

        ret_1m = (bar.Close - prev_close) / prev_close
        self.return_history.append(ret_1m)

        ret_5m = self._lookback_return(5)
        ret_30m = self._lookback_return(30)

        rolling_vol, skew, kurt = self._moment_stats()

        volume_norm, volume_z = self._volume_stats(bar.Volume)

        vwap = (self.vwap_pv / self.vwap_volume) if self.vwap_volume > 0 else bar.Close
        dist_vwap = (bar.Close - vwap) / vwap if vwap else 0.0

        norm_high, norm_low = self._normalized_range(bar.Close)

        ema_val = self.ema_indicator.Current.Value
        macd_hist = self.macd_indicator.Current.Value - self.macd_indicator.Signal.Current.Value
        rsi_val = self.rsi_indicator.Current.Value
        atr_val = self.atr_indicator.Current.Value
        sar_val = self.sar_indicator.Current.Value

        ema_slope = ((ema_val - self.prev_ema_val) / bar.Close) if self.prev_ema_val is not None and bar.Close else 0.0
        macd_slope = macd_hist - self.prev_macd_hist if self.prev_macd_hist is not None else 0.0
        rsi_slope = rsi_val - self.prev_rsi_val if self.prev_rsi_val is not None else 0.0
        atr_ratio = atr_val / bar.Close if bar.Close else 0.0
        sar_distance = (bar.Close - sar_val) / atr_val if atr_val else 0.0

        self.prev_ema_val = ema_val
        self.prev_macd_hist = macd_hist
        self.prev_rsi_val = rsi_val

        time_features = self._time_features(bar.EndTime)
        thirty_min_features = self._thirty_min_features()

        features = [
            ret_1m,
            ret_5m,
            ret_30m,
            rolling_vol,
            skew,
            kurt,
            dist_vwap,
            norm_high,
            norm_low,
            ema_slope,
            macd_slope,
            rsi_slope,
            atr_ratio,
            sar_distance,
            volume_norm,
            volume_z,
        ]
        features.extend(time_features)
        features.extend(thirty_min_features)
        return features if all(np.isfinite(f) for f in features) else None

    def _time_features(self, end_time: datetime) -> list[float]:
        et = end_time.astimezone(pytz.timezone("America/New_York"))
        minutes_since_midnight = et.hour * 60 + et.minute
        cycle = 2 * np.pi * minutes_since_midnight / (24 * 60)
        day_cycle = 2 * np.pi * et.weekday() / 7
        return [np.sin(cycle), np.cos(cycle), np.sin(day_cycle), np.cos(day_cycle)]

    def _thirty_min_features(self) -> list[float]:
        vector: list[float] = []
        prev_days = list(self.prev_day_30m)
        while len(prev_days) < self.PREV_30M_DAYS:
            prev_days.insert(0, [])
        prev_days = prev_days[-self.PREV_30M_DAYS:]
        for day_slots in prev_days:
            vector.extend(self._flatten_slots(day_slots))
        vector.extend(self._flatten_slots(self.current_day_30m))
        return vector

    def _flatten_slots(self, slots: list[Tuple[float, float, float, float, float, float, float]]) -> list[float]:
        result: list[float] = []
        slots_to_use = slots[-self.MAX_30M_SLOTS:]
        count = 0
        for slot in slots_to_use:
            result.extend(slot)
            count += 1
        while count < self.MAX_30M_SLOTS:
            result.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            count += 1
        return result

    def _lookback_return(self, minutes: int) -> float:
        if len(self.price_history) <= minutes:
            return 0.0
        past_price = list(self.price_history)[-minutes-1]
        return (self.price_history[-1] - past_price) / past_price if past_price else 0.0

    def _moment_stats(self) -> Tuple[float, float, float]:
        if len(self.return_history) < 10:
            return 0.0, 0.0, 0.0
        returns = np.array(self.return_history)
        mean = np.mean(returns)
        centered = returns - mean
        std = np.std(centered)
        if std == 0:
            return 0.0, 0.0, 0.0
        skew = np.mean((centered / std) ** 3)
        kurt = np.mean((centered / std) ** 4) - 3.0
        return std, skew, kurt

    def _volume_stats(self, current_volume: float) -> Tuple[float, float]:
        if not self.volume_history:
            return 1.0, 0.0
        volumes = np.array(self.volume_history, dtype=float)
        avg = np.mean(volumes)
        std = np.std(volumes)
        volume_norm = current_volume / avg if avg else 1.0
        volume_z = (current_volume - avg) / std if std else 0.0
        return volume_norm, volume_z

    def _normalized_range(self, price: float) -> Tuple[float, float]:
        if self.prev_day_high is None or self.prev_day_low is None:
            return 0.0, 0.0
        rng = self.prev_day_high - self.prev_day_low
        if rng <= 0:
            return 0.0, 0.0
        norm_high = (self.prev_day_high - price) / rng
        norm_low = (price - self.prev_day_low) / rng
        return norm_high, norm_low

    def _update_pending_labels(self, bar: TradeBar) -> None:
        while self.pending_labels and self.pending_labels[0]["target_time"] <= bar.EndTime:
            sample = self.pending_labels.popleft()
            future_ret = (bar.Close - sample["price"]) / sample["price"]
            label = 1 if future_ret > self.RETURN_THRESHOLD else 0
            self.training_features.append(sample["features"])
            self.training_labels.append(label)
            if len(self.training_labels) > self.MAX_TRAIN_SAMPLES:
                self.training_labels.pop(0)
                self.training_features.pop(0)

    def _maybe_train_model(self, force: bool = False) -> None:
        if not self.model or len(self.training_labels) < self.MIN_TRAIN_SAMPLES:
            return
        if not force and self.time.minute % 5 != 0:
            return
        X = np.array(self.training_features, dtype=float)
        y = np.array(self.training_labels, dtype=float)
        try:
            self.model.fit(X, y)
            self.model_trained = True
        except Exception as err:
            self.Log(f"XGBoost training failed: {err}")

    def _trade_from_prob(self, proba: float, bar: TradeBar) -> None:
        if self.time < self.training_unlock_time:
            return
        ny_time = self.time.astimezone(pytz.timezone("America/New_York"))
        local_time = ny_time.time()

        if self._is_global_market_closed(local_time):
            if self.contract_symbol and self.Portfolio[self.contract_symbol].Invested:
                self.Liquidate(self.contract_symbol, tag="Global session close")
            return

        if not self.TRADE_EXTENDED_HOURS:
            if local_time >= self.FLAT_ET:
                if self.contract_symbol and self.Portfolio[self.contract_symbol].Invested:
                    self.Liquidate(self.contract_symbol, tag="EOD Flatten")
                return
            if not self._is_regular_session(local_time):
                if self.contract_symbol and self.Portfolio[self.contract_symbol].Invested:
                    self.Liquidate(self.contract_symbol, tag="RTH only session close")
                return

        holding = self.Portfolio[self.contract_symbol].Quantity
        target_qty = holding

        if holding == 0:
            if proba >= self.PROB_LONG_ENTER:
                target_qty = self.CONTRACTS
            elif proba <= self.PROB_SHORT_ENTER:
                target_qty = -self.CONTRACTS
        elif holding > 0:
            if proba < self.PROB_LONG_EXIT:
                target_qty = 0
        elif holding < 0:
            if proba > self.PROB_SHORT_EXIT:
                target_qty = 0

        if target_qty == holding:
            return
        if target_qty == 0:
            self.Liquidate(self.contract_symbol, tag="Flat ML signal")
        else:
            self.MarketOrder(self.contract_symbol, target_qty - holding, tag=f"ML prob={proba:.2f}")

    # -------------- lifecycle -------------------
    def on_end_of_algorithm(self) -> None:
        self._remove_consolidators()
    def _process_primary_bar(self, bar: TradeBar, is_pretrain: bool = False) -> None:
        if not is_pretrain and self.IsWarmingUp:
            return
        self._reset_session_if_needed(bar.EndTime)
        self._update_vwap(bar)
        features = self._build_features(bar)
        self._update_pending_labels(bar)
        if not is_pretrain:
            self._maybe_train_model()
            if features and self.model_trained:
                proba = self.model.predict_proba(np.array(features).reshape(1, -1))[0][1]
                self.plot("Model", "ProbLong", proba)
                self._trade_from_prob(proba, bar)
        if features:
            target_time = bar.EndTime + timedelta(minutes=self.LABEL_HORIZON_MIN)
            self.pending_labels.append({
                "features": features,
                "price": bar.Close,
                "target_time": target_time
            })
