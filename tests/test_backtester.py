import unittest
import unittest
from yauber_backtester._backtester import Backtester

from yauber_backtester import Asset, Strategy, Account
from unittest import mock
import pandas as pd
import numpy as np


def make_rnd_asset(name):
    dt_index = pd.date_range('2016-01-01', '2018-01-01')
    ser = pd.Series(np.random.normal(size=len(dt_index)), index=dt_index).cumsum()
    asset_dict = {
        'ticker': f'RND_{name}',
        'quotes': pd.DataFrame({
            'o': ser,
            'h': ser,
            'l': ser,
            'c': ser,
            'exec': ser.shift(-1),
        }, index=ser.index),
    }

    return Asset(**asset_dict)


class TestStrategy(Strategy):
    def calculate(self, asset: Asset) -> pd.DataFrame:
        """
        Calculates main logic of the strategy, this method must return pd.DataFrame or None (if asset is filtered at all)
        This information is used by portfolio composition stage
        """
        # Simulate also NaNs
        return asset.quotes().rolling(20).mean()[['o', 'h', 'l', 'c', 'exec']]


class BacktesterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_universe = [
            make_rnd_asset('a1'),
            make_rnd_asset('a2'),
            make_rnd_asset('a3'),
        ]
        cls.strategy = TestStrategy()

    def test__process_metrics(self):
        bt = Backtester(self.asset_universe, self.strategy)
        df_all_metrics = bt._process_metrics()

        self.assertEqual(True, isinstance(df_all_metrics, pd.DataFrame))
        assert all(df_all_metrics.columns.levels[0] == self.asset_universe)
        assert all(df_all_metrics.columns.levels[1] == ['o', 'h', 'l', 'c', 'exec'])

    def test__process_metrics_wrong_dtypes(self):
        def calc_side(asset):
            df = asset.quotes().rolling(20).mean()[['o', 'h', 'l', 'c', 'exec']]
            df['test'] = 'A'
            return df
        smock = mock.MagicMock(self.strategy)
        smock.calculate.side_effect = calc_side
        bt = Backtester(self.asset_universe, smock)
        # raise ValueError(f"Couldn't convert {self.strategy}.calculate({asset}) result to float dtype, "
        self.assertRaises(ValueError, bt._process_metrics)

    def test__process_metrics_wrong_col_order(self):
        def calc_side(asset):
            if asset == "RND_a1":
                cols = ['o', 'h', 'l', 'c', 'exec']
            else:
                cols = ['exec', 'o', 'h', 'l', 'c']
            df = asset.quotes().rolling(20).mean()[cols]
            return df
        smock = mock.MagicMock(self.strategy)
        smock.calculate.side_effect = calc_side
        bt = Backtester(self.asset_universe, smock)
        #  raise ValueError(f"{self.strategy}.calculate() must return the same metrics and in the same order for each asset")
        self.assertRaises(ValueError, bt._process_metrics)

    def test__process_metrics_wrong_col_count(self):
        def calc_side(asset):
            if asset == "RND_a1":
                cols = ['o', 'h', 'l', 'c', 'exec']
            else:
                cols = ['o', 'h', 'l']
            df = asset.quotes().rolling(20).mean()[cols]
            return df
        smock = mock.MagicMock(self.strategy)
        smock.calculate.side_effect = calc_side
        bt = Backtester(self.asset_universe, smock)
        #  raise ValueError(f"{self.strategy}.calculate() must return the same metrics and in the same order for each asset")
        self.assertRaises(ValueError, bt._process_metrics)

    def test__run(self):
        def calc_side(asset):
            cols = ['o', 'h', 'l', 'c', 'exec']
            df = asset.quotes().rolling(20).mean()[cols]
            return df
        smock = mock.MagicMock(self.strategy)
        smock.calculate.side_effect = calc_side
        smock.compose_portfolio.return_value = {self.asset_universe[0]: 1}

        with mock.patch('yauber_backtester._account.Account._process_position') as mock_acc_process:
            bt = Backtester(self.asset_universe, smock)
            res = bt.run()

            self.assertEqual(True, smock.initialize.called)
            self.assertEqual(True, smock.compose_portfolio.called)
            self.assertEqual(True, mock_acc_process.called)
            self.assertEqual(True, isinstance(res, Account))

    def test__run_wrong_date_order_check(self):
        def calc_side(asset):
            cols = ['o', 'h', 'l', 'c', 'exec']
            df = asset.quotes().rolling(20).mean()[cols]
            return df
        smock = mock.MagicMock(self.strategy)
        smock.calculate.side_effect = calc_side
        smock.compose_portfolio.return_value = {self.asset_universe[0]: 1}

        bt = Backtester(self.asset_universe, smock)
        metrics_reversed = bt._process_metrics().sort_index(ascending=False)
        mock__process_metrics = mock.Mock()
        mock__process_metrics.return_value = metrics_reversed
        bt._process_metrics = mock__process_metrics
        # ValueError: Inconsistent datetime index order, quotes must be sorted in ascending order
        self.assertRaises(ValueError, bt.run)


if __name__ == '__main__':
    unittest.main()
