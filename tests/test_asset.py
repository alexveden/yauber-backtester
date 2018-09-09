import unittest
from yauber_backtester import Asset
import pandas as pd
import numpy as np
from unittest import mock


class AssetTestCase(unittest.TestCase):
    def setUp(self):
        self.quotes = pd.DataFrame(
            {
                'c': [1, 2, 3, 4, 5, 6],
                'exec': [2, 3, 4, 5, 6, 7],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )

    def test_init_and_internals(self):
        _asset_dict_bad = {
            'ticker': 'test_ticker',
            'quotes': [1, 2, 3],
        }
        self.assertRaises(ValueError, Asset, **_asset_dict_bad)

        q1 = pd.DataFrame(
            {
                'exec': [2, 3, 4, 5, 6, 7],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        q2 = pd.DataFrame(
            {
                'c': [1, 2, 3, 4, 5, 6],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        _asset_dict_bad = {
            'ticker': 'test_ticker',
            'quotes': q1,
        }
        self.assertRaises(ValueError, Asset, **_asset_dict_bad)

        _asset_dict_bad = {
            'ticker': 'test_ticker',
            'quotes': q2,
        }
        self.assertRaises(ValueError, Asset, **_asset_dict_bad)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes
        }
        a = Asset(**_asset_dict)

        self.assertEqual(a.ticker, 'test_ticker')
        self.assertEqual(id(a.quotes()), id(self.quotes))
        self.assertEqual(hash(a), hash(a.ticker))
        self.assertEqual(str(a), 'test_ticker')
        self.assertEqual(repr(a), 'Asset<test_ticker>')

    def test_get_prices(self):

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes.drop(self.quotes.index),
        }
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
        }
        a = Asset(**_asset_dict)

        # Requested quote is prior the data
        self.assertRaises(KeyError, a.get_prices, pd.Timestamp('2017-01-31'))

        # Valid case
        self.assertEqual((1, 2), a.get_prices(pd.Timestamp('2018-01-01')))

        # Handle data holes (return previous date data)
        self.assertEqual((3, 4), a.get_prices(pd.Timestamp('2018-01-04')))

        # Continue last quote if no data available anymore
        self.assertEqual((6, 7), a.get_prices(pd.Timestamp('2018-01-21')))

        # Check caching
        self.assertEqual(pd.Timestamp('2018-01-21'), a._cache_px_date)
        self.assertEqual((6, 7), a._cache_px_result)
        # Even empty quotes dataframe will no raise exception because of caching
        a._quotes = a._quotes.drop(a._quotes.index)
        self.assertEqual((6, 7), a.get_prices(pd.Timestamp('2018-01-21')))

    def test_get_pointvalue(self):

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
        }
        a = Asset(**_asset_dict)

        # Default point value is 1.0
        self.assertEqual(1, a.get_point_value(pd.Timestamp('2018-01-21')))

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 2.0
        }
        a = Asset(**_asset_dict)
        self.assertEqual(2, a.get_point_value(pd.Timestamp('2018-01-21')))

    def test_get_costs_zero(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
        }
        a = Asset(**_asset_dict)

        self.assertEqual((0, 0), a.get_costs(pd.Timestamp('2018-01-21'), 1))
        self.assertEqual((0, 0), a.get_costs(pd.Timestamp('2018-01-21'), -1))

    def test_get_costs_percent(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'percent',  # percent or dollar or dynamic
                'value': 0.001,     # 0.1%
            }
        }
        a = Asset(**_asset_dict)

        # Default point value is 1.0
        self.assertEqual((-0.002, -0.004), a.get_costs(pd.Timestamp('2018-01-01'), 2))
        self.assertEqual((-0.002, -0.004), a.get_costs(pd.Timestamp('2018-01-01'), -2))

        # Error checks
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'percent',  # percent or dollar or dynamic
                'value': [0.001],  # 0.1%
            }
        }
        # raise ValueError("'costs' value of 'percent' type must be a single float number")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'percent',  # percent or dollar or dynamic
                'value': -0.001,  # 0.1%
            }
        }
        # raise ValueError("'costs' value of 'percent' type must be positive")
        self.assertRaises(ValueError, Asset, **_asset_dict)


    def test_get_costs_dollar(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dollar',  # percent or dollar or dynamic
                'value': 3,
            }
        }
        a = Asset(**_asset_dict)

        # Default point value is 1.0
        self.assertEqual((-6, -6), a.get_costs(pd.Timestamp('2018-01-01'), 2))
        self.assertEqual((-6, -6), a.get_costs(pd.Timestamp('2018-01-01'), -2))

        # Error checks
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dollar',  # percent or dollar or dynamic
                'value': [3],
            }
        }
        # raise ValueError("'costs' value of 'percent' type must be a single float number")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dollar',  # percent or dollar or dynamic
                'value': -3,
            }
        }
        # raise ValueError("'costs' value of 'percent' type must be positive")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test_get_costs_dynamic(self):
        costs = pd.DataFrame(
            {
                'c': [1, 1, 1, 2, 2, 2],
                'exec': [-2, 3, 4, 5, 2, -3],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': costs,
            }
        }
        a = Asset(**_asset_dict)

        self.assertEqual((-2, -4), a.get_costs(pd.Timestamp('2018-01-01'), 2))
        self.assertEqual((-2, -4), a.get_costs(pd.Timestamp('2018-01-01'), -2))

        # Data holes
        self.assertEqual((-2, -8), a.get_costs(pd.Timestamp('2018-01-04'), 2))
        self.assertEqual((-2, -8), a.get_costs(pd.Timestamp('2018-01-04'), -2))

        self.assertRaises(KeyError, a.get_costs, pd.Timestamp('2017-01-04'), -2)

        #
        # Error checks
        #
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': [1, 2, 3],
            }
        }
        # raise ValueError("'costs' value of 'dynamic' type must be a Pandas.DataFrame with columns ['c', 'exec']")
        self.assertRaises(ValueError, Asset, **_asset_dict)
        #
        #
        #
        costs = pd.DataFrame(
            {
                'exec': [-2, 3, 4, 5, 2, -3],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': costs,
            }
        }
        # raise ValueError("'costs' value of 'dynamic' type must be a Pandas.DataFrame with columns ['c', 'exec']")
        self.assertRaises(ValueError, Asset, **_asset_dict)
        #
        #
        #
        costs = pd.DataFrame(
            {
                'c': [1, 1, 1, 2, 2, 2],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': costs,
            }
        }
        # raise ValueError("'costs' value of 'dynamic' type must be a Pandas.DataFrame with columns ['c', 'exec']")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        #
        # Inconsistent length
        #
        costs = pd.DataFrame(
            {
                'exec': [-2, 3, 4, 5, 2],
                'c': [1, 1, 1, 2, 2],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': costs,
            }
        }
        #raise ValueError("'costs' value of 'dynamic' dataframe must be the same length and index as quotes")
        self.assertRaises(ValueError, Asset, **_asset_dict)
        #
        # Incosistent index of quotes and costs dataframes
        #
        costs = pd.DataFrame(
            {
                'exec': [-2, 3, 4, 5, 2, -3],
                'c': [1, 1, 1, 2, 2, 2],
            },
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-10']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'dynamic',  # percent or dollar or dynamic
                'value': costs,
            }
        }
        # raise ValueError("'costs' value of 'dynamic' dataframe must be the same length and index as quotes")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test_get_costs_generic_errors(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'slkdfjslakjf;lsajkf',  # percent or dollar or dynamic
                'value': 3,
            }
        }
        # raise ValueError(f"Unknown costs type {costs_dict['type']}, only 'percent', 'dollar', 'dynamic' are supported")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        #
        # Costs must be a dict
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': []
        }
        # raise ValueError("'costs' in asset's kwargs must be a dict with {'type': ... and 'value': ... } keys")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        #
        # Missing 'type'
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'value': 3,
            }
        }
        # raise ValueError("'costs' in asset's kwargs must be a dict with {'type': ... and 'value': ... } keys")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        #
        # Missing 'value'
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'costs': {
                'type': 'percent',  # percent or dollar or dynamic
            }
        }
        # raise ValueError("'costs' in asset's kwargs must be a dict with {'type': ... and 'value': ... } keys")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test_calc_position_value(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100
        }
        a = Asset(**_asset_dict)

        # Valid case
        self.assertEqual(2 * 100 * 100, a.calc_position_value(pd.Timestamp('2018-01-01'), 100))
        self.assertEqual(2 * 100 * 100, a.calc_position_value(pd.Timestamp('2018-01-01'), -100))

    def test_get_margin_requirements(self):

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
        }
        a = Asset(**_asset_dict)

        # No margin settings - uses cash like
        self.assertEqual(2 * 100 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), 100))
        self.assertEqual(2 * 100 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), -100))

        #
        # % margin
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': 0.5
        }
        a = Asset(**_asset_dict)

        self.assertEqual(2 * 100 * 100 * 0.5, a.get_margin_requirements(pd.Timestamp('2018-01-01'), 100))
        self.assertEqual(2 * 100 * 100 * 0.5, a.get_margin_requirements(pd.Timestamp('2018-01-01'), -100))

        #
        # dollar margin
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': 500
        }
        a = Asset(**_asset_dict)

        self.assertEqual(500 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), 100))
        self.assertEqual(500 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), -100))

        #
        # dynamic margin
        #
        mgn = pd.Series([10, 10, 10, 20, 20, -20],
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': mgn,
        }
        a = Asset(**_asset_dict)

        self.assertEqual(10 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), 100))
        self.assertEqual(10 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-01'), -100))

        self.assertEqual(10 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-04'), 100))
        self.assertEqual(10 * 100, a.get_margin_requirements(pd.Timestamp('2018-01-04'), -100))
        #
        # Error checks
        #

        # raise KeyError(f'No margin found at {date}, margin range {self.margin.index[0]} - {self.margin.index[-1]}')
        self.assertRaises(KeyError, a.get_margin_requirements, pd.Timestamp('2017-01-04'), 100)
        # raise ValueError(f'Margin requirements for the asset {self} is negative at {date} value: {ser[-1]}')
        self.assertRaises(ValueError, a.get_margin_requirements, pd.Timestamp('2018-01-09'), 100)

    def test_init_margin_errors_checks(self):
        mgn = pd.Series([10, 10, 10, 20, 20, -20],
                        index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                                         '2018-01-07', '2018-01-08', '2018-01-10']]
                        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': mgn,
        }
        # ValueError: 'margin' pd.Series must have the same length and index as quotes
        self.assertRaises(ValueError, Asset, **_asset_dict)



        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': -1,
        }
        # raise ValueError("'margin' must be >= 0")
        self.assertRaises(ValueError, Asset, **_asset_dict)



        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
            'margin': 'wrongtype',
        }
        # raise ValueError("'margin' unsupported type of asset margin, it must be pd.Series or float")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test_calc_dollar_pnl(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100
        }
        a = Asset(**_asset_dict)

        # Valid case
        self.assertEqual(1 * 100 * 100, a.calc_dollar_pnl(pd.Timestamp('2018-01-01'), 1, 2, 100))
        self.assertEqual(1 * 100 * -100, a.calc_dollar_pnl(pd.Timestamp('2018-01-01'), 1, 2, -100))

    def test_is_synthetic(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100
        }
        a = Asset(**_asset_dict)

        # Valid case
        self.assertEqual(False, a.is_synthetic)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'is_synthetic': True,
            'point_value': 100
        }
        a = Asset(**_asset_dict)

        # Valid case
        self.assertEqual(True, a.is_synthetic)

    def test_legs(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100
        }
        a = Asset(**_asset_dict)
        self.assertEqual({'test_ticker': 1.0}, a.legs)

        # Multileg asset
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'legs': {
                'a': 1,
                'b': -1,
            }
        }
        a = Asset(**_asset_dict)
        self.assertEqual({'a': 1, 'b': -1}, a.legs)

        #
        # Error checks
        #
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'legs': {
                'a': pd.Series(),
                'b': -1,
            }
        }
        # ValueError: Asset 'legs' values must be numbers, got <class 'pandas.core.series.Series'>
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'legs': {
                'a': -1,
                a: -1,
            }
        }
        # raise ValueError(f"Asset 'legs' keys must be strings, got {type(k)}")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'legs': []
        }
        # raise ValueError("Asset 'legs' must be a dictionary of {<ticker_string>: <qty_float>}")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test_get_point_value_dynamic(self):

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 100,
        }
        a = Asset(**_asset_dict)

        # No margin settings - uses cash like
        self.assertEqual(100, a.get_point_value(pd.Timestamp('2018-01-01')))


        #
        # dynamic
        #
        pv = pd.Series([10, 10, 10, 20, 0, -20],
            index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                             '2018-01-07', '2018-01-08', '2018-01-09']]
        )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': pv,
        }
        a = Asset(**_asset_dict)
        self.assertEqual(10, a.get_point_value(pd.Timestamp('2018-01-01')))
        self.assertEqual(10, a.get_point_value(pd.Timestamp('2018-01-04')))
        #
        # Error checks
        #
        # raise KeyError(f'No point value found at {date}, range {self._point_value.index[0]} - {self._point_value.index[-1]}')
        self.assertRaises(KeyError, a.get_point_value, pd.Timestamp('2017-01-01'))
        # raise ValueError(f'Point value for the asset {self} is <= 0 at {date} value: {ser[-1]}')
        self.assertRaises(ValueError, a.get_point_value, pd.Timestamp('2018-01-09'))
        self.assertRaises(ValueError, a.get_point_value, pd.Timestamp('2018-01-08'))

        #
        # Caching
        #
        self.assertEqual(10, a.get_point_value(pd.Timestamp('2018-01-01')))
        a._point_value = None
        # This should use cache
        self.assertEqual(10, a.get_point_value(pd.Timestamp('2018-01-01')))

        #
        # Initialization errors
        #
        pv = pd.Series([10, 10, 10, 20, 0, -20],
                       index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                                        '2018-01-07', '2018-01-08', '2018-01-10']]
                       )
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': pv,
        }
        # ValueError: 'point_value' pd.Series must have the same length and index as quotes
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 0,
        }
        # raise ValueError("'pointvalue' must be > 0")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': -1,
        }
        # raise ValueError("'pointvalue' must be > 0")
        self.assertRaises(ValueError, Asset, **_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
            'point_value': 'bad',
        }
        # raise ValueError(f"'point_value' unsupported type, it must be pd.Series or float, got {type(self._point_value)}")
        self.assertRaises(ValueError, Asset, **_asset_dict)

    def test__eq__(self):
        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
        }
        a = Asset(**_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker',
            'quotes': self.quotes,
        }
        a2 = Asset(**_asset_dict)

        _asset_dict = {
            'ticker': 'test_ticker2',
            'quotes': self.quotes,
        }
        a3 = Asset(**_asset_dict)

        self.assertEqual(False, a == None)
        self.assertEqual(False, a == 'test')
        self.assertEqual(True, a == a2)
        self.assertEqual(False, a == a3)

if __name__ == '__main__':
    unittest.main()
