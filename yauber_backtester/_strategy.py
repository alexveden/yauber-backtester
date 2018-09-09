from typing import Tuple
import pandas as pd
import numpy as np
from ._asset import Asset
from ._account import Account
from datetime import datetime


class Strategy:
    """
    Generic strategy class with portfolio management functionality
    """
    name = 'BaseStrategy'

    def __init__(self, **strategy_context):
        self.context = strategy_context
        """Strategy initial dictionary"""

        self.params = self.context.get('params', {})
        """Strategy default params dict"""

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Strategy<{self.name}>"

    def initialize(self):
        """
        Initialize and reset strategy cache (if any)
        :return:
        """
        pass

    def calculate(self, asset: Asset) -> pd.DataFrame:
        """
        Calculates main logic of the strategy, this method must return pd.DataFrame or None (if asset is filtered at all)
        This information is used by portfolio composition stage
        """
        return None

    def compose_portfolio(self, date: datetime, account: Account, asset_metrics: pd.DataFrame) -> dict:
        """
        Returns a dictionary of portfolio composition at specific 'date'
        :param date: analysis date
        :param account: actual account at the previous date (see. Account class interface)
        :param asset_metrics: composite pd.DataFrame returned by self.calculate() method at 'date', with index as asset, and columns as metrics names

        :return: dictionary of  {asset_class_instance: float_opened_quantity, ... }
        Notes:
        This method is about managing portfolios, you can implement asset ranking based on asset_metrics, or Money Management strategy.
        This method permits opening fractional position sizes, or opening positions with negative capital. You should explicitly manage
        all possible issues with portfolio composition and perform all checks in this method.

        You can use information returned by asset.info(date) to get all information about the asset. The backtester engine will use this
        information to retrieve execution price and costs, as well.
        """
        return {}