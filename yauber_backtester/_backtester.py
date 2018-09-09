from typing import List
import pandas as pd
import numpy as np
from ._asset import Asset
from ._strategy import Strategy
from ._account import Account
import numba
from math import nan


@numba.jit(nopython=True)
def _unstack(values, n_assets, n_fields):  # pragma: no cover
    result = np.full((n_assets, n_fields), nan)

    for i in range(n_assets):
        for j in range(n_fields):
            result[i, j] = values[i * n_fields + j]
    return result


class Backtester:
    """
    Generic portfolio backtester
    """

    def __init__(self, asset_universe: List[Asset], strategy: Strategy, **kwargs):
        self.asset_universe = asset_universe
        self.strategy = strategy
        self.kwargs = kwargs

    def _process_metrics(self):
        """
        Collects metrics for all assets in universe and prepares dataset for portfolio composition stage
        :return:
        """
        # Step 1: launch self.strategy.calculate() for every asset in the universe and produce asset metrics
        asset_metrics_all = {}
        col_names = None

        for asset in self.asset_universe:
            _res = self.strategy.calculate(asset)
            try:
                _res = _res.astype(np.float, copy=False)
            except:
                raise ValueError(f"Couldn't convert {self.strategy}.calculate({asset}) result to float dtype, "
                                 f"calculate() method must return a pd.DataFrame with numbers (int, float, bool), no objects or strings are allowed!")

            if col_names is None:
                col_names = _res.columns

            if len(col_names) != len(_res.columns) or not np.all(col_names == _res.columns):
                raise ValueError(f"{self.strategy}.calculate() must return the same metrics and in the same order for each asset")

            asset_metrics_all[asset] = _res

        # Step 2: Join and align all asset metrics into the single dataset
        df_all_metrics = pd.concat(asset_metrics_all.values(), keys=asset_metrics_all.keys(), axis=1, copy=False)

        # Make sure that pandas haven't reordered the assets and columns order after concatenation
        assert all(df_all_metrics.columns.levels[0] == self.asset_universe)
        assert all(df_all_metrics.columns.levels[1] == col_names)

        return df_all_metrics

        idx = pd.Index([1, 2])
        idx.unique()

    def run(self) -> Account:
        """
        Runs portfolio backtesting for all assets in universe
        :return: Account class
        """
        # Initialize and reset strategy cache (if any)
        self.strategy.initialize()

        # Get asset universe combined metrics
        df_all_metrics = self._process_metrics()

        acc = Account(buffer_len=len(df_all_metrics),
                      name=self.kwargs.get('acc_name', repr(self.strategy)),
                      initial_capital=self.kwargs.get('acc_initial_capital', 0),
                      )

        last_dt = None

        n_assets = len(df_all_metrics.columns.levels[0])
        n_cols = len(df_all_metrics.columns.levels[1])

        for dt, row in df_all_metrics.iterrows():
            # Perform some sanity checks
            if last_dt is not None:
                if dt <= last_dt:
                    raise ValueError("Inconsistent datetime index order, quotes must be sorted in ascending order")

            # Get metrics for specific date and unstack them to the dataframe
            # Fast unstacking to dataframe of metrics
            _metrics = pd.DataFrame(_unstack(row.values, n_assets, n_cols),
                                    index=df_all_metrics.columns.levels[0],
                                    columns=df_all_metrics.columns.levels[1])

            # Call strategy.compose_portfolio()
            new_pos = self.strategy.compose_portfolio(dt, acc, _metrics)

            # Process new position
            acc._process_position(dt, new_pos)

            last_dt = dt

        if len(df_all_metrics) > 0:
            # Finally make sure that fast _unstack function values are exactly the same to pandas.unstack()
            assert np.allclose(row.unstack().values, _metrics.values, equal_nan=True)

        return acc
