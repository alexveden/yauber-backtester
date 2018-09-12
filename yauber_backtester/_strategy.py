from typing import Tuple
import pandas as pd
import numpy as np
from ._asset import Asset
from ._account import Account
from ._containers import MFrame
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

    def compose_portfolio(self, date: datetime, account: Account, mf: MFrame) -> dict:
        """
        Returns a dictionary of portfolio composition at specific 'date'
        :param date: analysis date
        :param account: actual account at the previous date (see. Account class interface)
        :param mf: composite of metrics returned by self.calculate() method at 'date'

        :return: dictionary of  {asset_class_instance: float_opened_quantity, ... }

        -----------
        mf - cheat sheet
        -----------
        mf.assets - list of all assets
        mf.columns - list of all columns / metrics
        mf['metric_name'] - get metric 'metric_name' numpy array across all assets
        for asset, row in mf.items():  - iterate over all assets and rows, you can use row['metric_name'] too
        mf.get_at('asset_ticker', 'metric_name') - get scalar value of 'metric_name' for asset 'asset_ticker'
        mf.get_asset('asset_ticker') - get asset object by ticker name
        mf.get_filtered((mf['some_metric'] > 0) & (mf['another_metric'] == 1), sort_by_col='another_metric'[or None]) - get filtered and sorted data
        for (asset, m_data) in zip(*mf.get_filtered(_cond, sort_by_col='ma200')): - iterate over filtered and sorted results
        filtered_assets, filtere_data  = mf.get_filtered(..some condition..) - get filtered asset list and metrics
        mf.as_dataframe() - converts MFrame to Pandas.DataFrame. Warning: calculations might become much slower!
        """
        return {}