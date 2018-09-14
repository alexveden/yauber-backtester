from typing import List
import pandas as pd
import numpy as np
from ._asset import Asset
from ._strategy import Strategy
from ._account import Account
from ._containers import MFrame


class Backtester:
    """
    Generic portfolio backtester
    """
    @staticmethod
    def _process_metrics(strategy, asset_universe):
        """
        Collects metrics for all assets in universe and prepares dataset for portfolio composition stage
        :return:
        """
        # Step 1: launch self.strategy.calculate() for every asset in the universe and produce asset metrics
        asset_metrics_all = {}
        col_names = None

        for asset in asset_universe:
            _res = strategy.calculate(asset)
            try:
                _res = _res.astype(np.float, copy=False)
            except:
                raise ValueError(f"Couldn't convert {strategy}.calculate({asset}) result to float dtype, "
                                 f"calculate() method must return a pd.DataFrame with numbers (int, float, bool), no objects or strings are allowed!")

            if col_names is None:
                col_names = _res.columns

            if len(col_names) != len(_res.columns) or not np.all(col_names == _res.columns):
                raise ValueError(f"{strategy}.calculate() must return the same metrics and in the same order for each asset")

            asset_metrics_all[asset] = _res

        # Step 2: Join and align all asset metrics into the single dataset
        df_all_metrics = pd.concat(asset_metrics_all.values(), keys=asset_metrics_all.keys(), axis=1, copy=False)

        # Make sure that pandas haven't reordered the assets and columns order after concatenation
        assert all(df_all_metrics.columns.levels[0] == asset_universe)
        assert all(df_all_metrics.columns.levels[1] == col_names)

        return df_all_metrics

    @staticmethod
    def run(strategy: Strategy, asset_universe: List[Asset], **kwargs) -> Account:
        """
        Runs portfolio backtesting for all assets in universe
        :param strategy: Strategy class instance
        :param asset_universe: list of assets
        :param kwargs:
            - 'acc_name' - resulting account name (by default: uses strategy name)
            - 'acc_initial_capital' - initial capital (default: 0)
        :return: Account class
        """
        # Initialize and reset strategy cache (if any)
        strategy.initialize()

        # Get asset universe combined metrics
        df_all_metrics = Backtester._process_metrics(strategy, asset_universe)

        acc = Account(buffer_len=len(df_all_metrics),
                      name=kwargs.get('acc_name', str(strategy)),
                      initial_capital=kwargs.get('acc_initial_capital', 0),
                      )

        last_dt = None

        # Setting vals / dt_idx in sake of performance
        vals = df_all_metrics.values
        dt_idx = df_all_metrics.index
        mframe = MFrame(assets=df_all_metrics.columns.levels[0],
                        columns=df_all_metrics.columns.levels[1])

        for i in range(df_all_metrics.values.shape[0]):
            row = vals[i]
            dt = dt_idx[i]

            # Perform some sanity checks
            if last_dt is not None:
                if dt <= last_dt:
                    raise ValueError("Inconsistent datetime index order, quotes must be sorted in ascending order")

            # Get metrics for specific date and unstack them to the dataframe
            # Fast unstacking to dataframe of metrics
            mframe._fill(row)

            # Call strategy.compose_portfolio()
            new_pos = strategy.compose_portfolio(dt, acc, mframe)

            # Process new position
            acc._process_position(dt, new_pos)

            last_dt = dt

        return acc
