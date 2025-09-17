from AlgorithmImports import *
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from QuantConnect.Indicators import RollingWindow
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Data.Consolidators import TradeBarConsolidator
from QuantConnect import Resolution, Field
from QuantConnect.Algorithm.Framework.Selection import CoarseFundamentalUniverseSelectionModel


from datetime import timedelta

class MacdDecisionTree(QCAlgorithm):

  def initialize(self) -> None:
    # Backtest period and cash
    self.set_start_date(2010, 1, 1)  
    self.set_end_date(2024, 9, 1)
    self.set_cash(100_000)  
    self.capital_allocation = 0.05
    self.trailing_stop_percent = 0.40
    self.max_value = 1
    self.symbol_data = {}
    
    self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
    self.spy_sma = self.sma(self.spy, 200, Resolution.DAILY)
    
    self.universe_settings.resolution = Resolution.DAILY
    self.set_universe_selection(CoarseFundamentalUniverseSelectionModel(self.create_universes, self.universe_settings))
    
    self.set_warm_up(timedelta(days=200))
    self.initial_filled_prices = {}
    self.next_rebalance = self.time
    self.next_model_train = self.time
    
    self.model = DecisionTreeClassifier(max_depth=5)
    self.scaler = StandardScaler()
    self.is_model_trained = False
    
  def create_universes(self, coarse):
    if self.time < self.next_rebalance:
      return Universe.UNCHANGED
    self.next_rebalance = self.time + timedelta(days=30)
    
    sorted_by_dollar_volume = sorted(
      [x for x in coarse if x.HasFundamentalData],
      key=lambda x: x.dollar_volume,
      reverse=True
    )
    
    return [x.Symbol for x in sorted_by_dollar_volume[:100]]
    
  def on_security_changed(self, changes):
    for security in changes.added_securities:
      symbol = security.symbol
      self.symbol_data[symbol] = SymbolData(self, symbol)
    
    for security in changes.removed_securities:
      symbol = security.symbol
      if symbol in self.symbol_data:
        self.symbol_data.pop(symbol)
      if symbol in self.initial_filled_prices:
        self.initial_filled_prices.pop(symbol)
      
  def on_data(self, data):
    if self.is_warming_up:
      return
    
    if self.time >= self.next_model_train and len(self.symbol_data) > 0:
      self.train_model()
      self.next_model_train = self.time + timedelta(days=30)
    
    if not self.is_model_trained:
      return
    
    if self.securities[self.spy].price <= self.spy_sma.current.value:
      self.liquidate()
    
    invested = [x.symbol for x in self.portfolio.get_values() if x.invested]
    
    # Capital allocation check - prevent over-allocation and zero quantity errors
    if len(invested) >= (self.max_value / self.capital_allocation):
      return
    
    for symbol, symbol_data in self.symbol_data.items():
      macd_hist = symbol_data.macd.current.value
      features = self.get_features(symbol_data)
      if features is not None:
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        
        if (
            macd_hist > 0 
            and self.securities[self.spy].price <= self.spy_sma.current.value
            and symbol_data.fast_ema.current.value > symbol_data.slow_ema.current.value
            and prediction == 1
          ):
          
            if symbol not in self.initial_filled_prices:
              quantity = self.calculate_order_quantity(symbol, self.capital_allocation)
              if quantity > 0:
                order = self.market_order(symbol, quantity)
                if order.status == OrderStatus.FILLED:
                  self.initial_filled_prices[symbol] = order.average_fill_price
                  symbol_data.entry_price = order.average_fill_price
                  symbol_data.stop_loss_price = order.average_fill_price * (1 - self.trailing_stop_percent)

  def train_model(self):
    X = []
    y = []
    
    for symbol, symbol_data in self.symbol_data.items():
      if symbol_data.macd.current.is_ready:
        features = self.get_features(symbol_data)
        if features is not None:
          X.append(features)
          y.append(1)
    
    if len(X) > 0 and len(X) == len(y):
      X = self.scaler.fit_transform(X)
      self.model.fit(X, y)
      self.is_model_trained = True
      self.debug("Model trained successfully")
      
  def get_features(self, symbol_data):
    result = None
    if symbol_data.macd.current.is_ready and symbol_data.volume_window.is_ready:
      vols = list(symbol_data.volume_window)  # newest at index 0
      sum30 = sum(vols[:30])
      sum60 = sum(vols)
      volume_ratio_30_60 = sum30 / sum60 if sum60 > 0 else 0

      result = [
        symbol_data.macd.current.value,
        symbol_data.fast_ema.current.value,
        symbol_data.slow_ema.current.value,
        symbol_data.rsi.current.value,
        symbol_data.roc.current.value,
        volume_ratio_30_60
      ]
    return result

class SymbolData:
  def __init__(self, algorithm: QCAlgorithm, symbol: Symbol):
      self.symbol = symbol
      self.algorithm = algorithm

      # Indicators
      self.macd     = algorithm.macd(symbol, 12, 26, 9, MovingAverageType.EXPONENTIAL, Resolution.DAILY, Field.CLOSE)
      self.fast_ema = algorithm.ema(symbol, 13, Resolution.DAILY, Field.CLOSE)
      self.slow_ema = algorithm.ema(symbol, 26, Resolution.DAILY, Field.CLOSE)
      self.rsi      = algorithm.rsi(symbol, 14, MovingAverageType.EXPONENTIAL, Resolution.DAILY, Field.CLOSE)
      self.roc      = algorithm.roc(symbol, 10, Resolution.DAILY, Field.CLOSE)
      self.volume_window = RollingWindow[float](60)  

      # Wire up a daily TradeBar consolidator to fill the volume window
      daily_consolidator = TradeBarConsolidator(timedelta(days=1))
      algorithm.subscription_manager.add_consolidator(symbol, daily_consolidator)
      daily_consolidator.DataConsolidated += self._on_data_consolidated

      # ── OPTIONAL: continue warming up your other indicators via consolidator
      #    (you can remove this if your indicators subscribe automatically)
      ind_consolidator = algorithm.resolve_consolidator(symbol, Resolution.DAILY)
      for ind in [self.macd, self.fast_ema, self.slow_ema, self.rsi, self.roc]:
          algorithm.subscription_manager.add_consolidator(symbol, ind_consolidator)
          ind_consolidator.DataConsolidated += ind.update

      # ── TRADE MANAGEMENT PLACEHOLDERS ────────────────────────
      self.entry_price     = None
      self.stop_loss_price = None

  def _on_data_consolidated(self, sender, bar: TradeBar):
      """Callback for daily bars → push volume into our window."""
      self.volume_window.add(bar.volume)