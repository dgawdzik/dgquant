from AlgorithmImports import *

# This algorithm is designed to troubleshoot data availability on Quant Connect or locally.
# The script was given as part of support ticket #2025-04-22-0001, where the user reported missing AAPL data around April 22, 2025.
class TroubleshootAAPLDataAlgorithm(QCAlgorithm):
    def initialize(self) -> None:
        # Set extended date range to encompass April 22, 2025 more fully
        # Going further back to ensure map files and any needed splits/dividends data are available
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 5, 1)
        self.set_cash(100000)

        # Add AAPL with multiple resolutions for troubleshooting:
        # - Primary: Minute
        # - Alternate: Daily
        # Note: removed explicit market parameter to use default for US equities
        self._aapl_symbol_minute = self.add_equity("AAPL", Resolution.MINUTE).symbol
        self._aapl_symbol_daily = self.add_equity("AAPL", Resolution.DAILY).symbol
        self.debug("AAPL (minute and daily) securities added.")

        # Prepare date references around April 22, 2025
        requested_date = datetime(2025, 4, 22)
        start_check = requested_date - timedelta(days=1)
        end_check = requested_date + timedelta(days=1)
        self.debug(f"Checking data existence from {start_check} to {end_check}...")

        # Attempt retrieving data with different resolutions to confirm availability
        # History (minute-level)
        minute_history = self.history(self._aapl_symbol_minute, start_check, end_check, Resolution.MINUTE)
        if minute_history.empty:
            self.debug("No MINUTE data found for AAPL around April 22, 2025.")
        else:
            self.debug(f"Minute data retrieved: {len(minute_history)} rows.")

        # History (daily-level)
        daily_history = self.history(self._aapl_symbol_daily, start_check, end_check, Resolution.DAILY)
        if daily_history.empty:
            self.debug("No DAILY data found for AAPL around April 22, 2025.")
        else:
            self.debug(f"Daily data retrieved: {len(daily_history)} rows.")

        # Retrieve a broader multi-day window of daily data leading up to April 22, 2025
        multi_day_start = requested_date - timedelta(days=5)
        multi_day_end = requested_date
        broad_history = self.history(self._aapl_symbol_daily, multi_day_start, multi_day_end, Resolution.DAILY)
        if broad_history.empty:
            self.debug("No multi-day (5-day) DAILY data leading up to April 22, 2025.")
        else:
            self.debug(f"Found {len(broad_history)} rows of daily data from {multi_day_start} to {multi_day_end}.")

        self.debug("Initialization complete. Data availability checks finished.")

    def on_data(self, data: Slice) -> None:
        # No trading logic for this troubleshooting example
        pass