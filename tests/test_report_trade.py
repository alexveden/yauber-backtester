import unittest
from yauber_backtester._report import Trade, TRADE_KEYS
import pandas as pd
import numpy as np

class TradeTestCase(unittest.TestCase):
    def test_init(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5,
            'context': ('context',)
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)

        self.assertEqual(t._is_closed, False)
        self.assertEqual(t._entry_date, dt)
        self.assertEqual(t._side, 1)
        self.assertEqual(t._entry_qty, 1)
        self.assertEqual(t._entry_value, 1 * 2)
        self.assertEqual(t._exit_qty, 0)
        self.assertEqual(t._exit_value, 0)
        self.assertEqual(t._costs, -0.2)
        self.assertEqual(t._n_transations, 1)
        self.assertEqual(t._pnl, 3.5)
        self.assertEqual(t._qty, 1)
        self.assertEqual(t._context, ('context',))

    def test_init_no_context(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5,
            'context': None,
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)

        self.assertEqual(t._is_closed, False)
        self.assertEqual(t._entry_date, dt)
        self.assertEqual(t._side, 1)
        self.assertEqual(t._entry_qty, 1)
        self.assertEqual(t._entry_value, 1 * 2)
        self.assertEqual(t._exit_qty, 0)
        self.assertEqual(t._exit_value, 0)
        self.assertEqual(t._costs, -0.2)
        self.assertEqual(t._n_transations, 1)
        self.assertEqual(t._pnl, 3.5)
        self.assertEqual(t._qty, 1)
        self.assertEqual(np.isnan(t._context), True)

    def test_close(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)

        trans2 = {
            'asset': 'test',
            'position_action': -1,
            'qty': -1,
            'price_close': 1,
            'price_exec': 4,
            'costs_close': -0.2,
            'costs_exec': -0.4,
            'pnl_close': 1.4,
            'pnl_execution': 1.5
        }
        dt2 = pd.Timestamp('2017-01-02')
        t.add_transaction(dt2, trans2)

        self.assertEqual(t._is_closed, True)
        self.assertEqual(t._entry_date, dt)
        self.assertEqual(t._side, 1)
        self.assertEqual(t._entry_qty, 1)
        self.assertEqual(t._entry_value, 1 * 2)
        self.assertEqual(t._exit_qty, 1)
        self.assertEqual(t._exit_value, 1 * 4)
        self.assertEqual(t._costs, -0.2 + -0.4)
        self.assertEqual(t._n_transations, 2)
        self.assertEqual(t._pnl, 3.5 + 1.5)
        self.assertEqual(t._qty, 0)

    def test_increase(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)

        trans2 = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 4,
            'costs_close': -0.2,
            'costs_exec': -0.4,
            'pnl_close': 1.4,
            'pnl_execution': 1.5
        }
        dt2 = pd.Timestamp('2017-01-02')
        t.add_transaction(dt2, trans2)

        self.assertEqual(t._is_closed, False)
        self.assertEqual(t._entry_date, dt)
        self.assertEqual(t._side, 1)
        self.assertEqual(t._entry_qty, 2)
        self.assertEqual(t._entry_value, 1 * 2 + 4 * 1)
        self.assertEqual(t._exit_qty, 0)
        self.assertEqual(t._exit_value, 0)
        self.assertEqual(t._costs, -0.2 + -0.4)
        self.assertEqual(t._n_transations, 2)
        self.assertEqual(t._pnl, 3.5 + 1.5)
        self.assertEqual(t._qty, 2)

    def test_increase_and_reverse(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)

        trans2 = {
            'asset': 'test',
            'position_action': -1,
            'qty': -2,
            'price_close': 1,
            'price_exec': 4,
            'costs_close': -0.2,
            'costs_exec': -0.4,
            'pnl_close': 1.4,
            'pnl_execution': 1.5
        }
        dt2 = pd.Timestamp('2017-01-02')
        self.assertRaises(AssertionError, t.add_transaction, dt2, trans2)

    def test_as_tuple_and_keys(self):
        trans = {
            'asset': 'test',
            'position_action': 1,
            'qty': 1,
            'price_close': 1,
            'price_exec': 2,
            'costs_close': -0.5,
            'costs_exec': -0.2,
            'pnl_close': 2.4,
            'pnl_execution': 3.5
        }
        dt = pd.Timestamp('2017-01-01')
        t = Trade(dt, trans)
        (
            _asset,
            _entry_date,
            _exit_date,
            _side,
            _n_transations,
            entry_avg_px,
            exit_avg_px,
            _entry_qty,
            _exit_qty,
            _pnl,
            _pnl_perc,
            _costs,
            _ctx
        ) = t.as_tuple()

        self.assertEqual(_asset, 'test')
        self.assertEqual(_entry_date, dt)
        self.assertEqual(_exit_date, dt)
        self.assertEqual(_side, 1)
        self.assertEqual(_n_transations, 1)
        self.assertEqual(entry_avg_px, 2)
        self.assertEqual(True, np.allclose([exit_avg_px], [np.nan], equal_nan=True))
        self.assertEqual(_entry_qty, 1)
        self.assertEqual(_exit_qty, 0)
        self.assertEqual(_pnl, 3.5)
        #self.assertEqual(_pnl_perc, (exit_avg_px / entry_avg_px - 1) * 1)
        self.assertEqual(_costs, -0.2)
        self.assertEqual(True, np.isnan(_ctx))

        # Trade keys
        keys = (
        'asset', 'date_entry', 'date_exit', 'side', 'n_transactions', 'wavg_price_entered', 'wavg_price_exited',
        'qty_entered', 'qty_exited', 'pnl', 'pnl_perc', 'costs', 'context')
        self.assertEqual(TRADE_KEYS, keys)

if __name__ == '__main__':
    unittest.main()
