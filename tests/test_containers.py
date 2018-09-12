import unittest
from yauber_backtester._containers import MFrame, _unstack, PositionInfo, RowTuple
from yauber_backtester import Backtester, Asset
from .test_backtester import make_rnd_asset, TestStrategy
import pandas as pd
import numpy as np


class ContainersTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_universe = [
            make_rnd_asset('a1'),
            make_rnd_asset('a2'),
            make_rnd_asset('a3'),
        ]
        cls.strategy = TestStrategy()

    def test__unstack(self):
        bt = Backtester(self.asset_universe, self.strategy)
        df_all_metrics = bt._process_metrics()

        n_assets = len(df_all_metrics.columns.levels[0])
        n_cols = len(df_all_metrics.columns.levels[1])
        for dt, row in df_all_metrics.iterrows():
            # Get metrics for specific date and unstack them to the dataframe
            # Fast unstacking to dataframe of metrics
            _metrics = pd.DataFrame(_unstack(row.values, n_assets, n_cols),
                                    index=df_all_metrics.columns.levels[0],
                                    columns=df_all_metrics.columns.levels[1])

            assert np.allclose(row.unstack().values, _metrics.values, equal_nan=True)

    def test_mframe__init(self):
        mf = MFrame(assets=self.asset_universe, columns=['d', 'e'])

        self.assertEqual((3,2), mf.shape)
        self.assertEqual(True, all(self.asset_universe == mf.assets))
        self.assertEqual(('d', 'e'), mf.columns)

        df = pd.DataFrame([
            [1, 2],
            [10, 20],
            [-1, -2],
        ], index=self.asset_universe, columns=['d', 'e'])

        _stacked = df.stack()
        mf._fill(_stacked.values)

        self.assertEqual(True, (mf._data == df.values).all())
        self.assertEqual(True, ([1, 10, -1] == mf['d']).all())
        self.assertEqual(True, ([2, 20, -2] == mf['e']).all())

        self.assertEqual(20, mf.get_at('RND_a2', 'e'))
        self.assertEqual(20, mf.get_at(mf.assets[1], 'e'))

        self.assertEqual(mf.assets[1], mf.get_asset('RND_a2'))
        self.assertEqual(True, isinstance(mf.get_asset('RND_a2'), Asset))

        flt_assets, flt_val = mf.get_filtered(mf['e'] < 20)
        self.assertEqual(2, len(flt_assets))
        self.assertEqual((2, 2), flt_val.shape)
        self.assertEqual(True, (flt_val == (df[df['e'] < 20].values)).all())

        # Filtered and sorted
        flt_assets, flt_val = mf.get_filtered(mf['e'] < 20, sort_by_col='e')
        self.assertEqual(2, len(flt_assets))
        self.assertEqual((2, 2), flt_val.shape)
        self.assertEqual(True, (flt_val == (df[df['e'] < 20].sort_values('e').values)).all())
        self.assertEqual(True, np.all(np.array([[-1, -2], [1, 2]])  == flt_val))
        self.assertEqual(True, np.all(mf.assets.take([2, 0]) == flt_assets))

        # Items
        for a, r in mf.items():
            self.assertEqual(True, isinstance(a, Asset))
            self.assertEqual(True, isinstance(r, RowTuple))

            self.assertEqual(r['e'], mf.get_at(a, 'e'))
            self.assertEqual(r['d'], mf.get_at(a, 'd'))

    def test_position_info(self):
        p = PositionInfo(self.asset_universe[0], -1)
        self.assertEqual(p.asset, self.asset_universe[0])
        self.assertEqual(p.qty, -1)

        self.assertEqual(str(p), f"{self.asset_universe[0]} x {-1}")
        self.assertEqual(str(p), repr(p))

if __name__ == '__main__':
    unittest.main()
