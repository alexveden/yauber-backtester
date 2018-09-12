import unittest
from yauber_backtester._account import Account
from yauber_backtester._asset import Asset
from unittest import mock
import pandas as pd
import numpy as np

class AccountTestCase(unittest.TestCase):
    def setUp(self):
        self.asset1 = Asset(**{
            'ticker': 'A',
            'quotes': pd.DataFrame(
                                        {
                                               'c': [1, 2, 3, 4, 5, 6],
                                            'exec': [2, 3, 4, 5, 6, 7],
                                        },
                                        index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                                                         '2018-01-07', '2018-01-08', '2018-01-09']]
                                    ),
            'costs': {
                'type': 'percent',
                'value': 0.5,
            }
        })

        self.asset2 = Asset(**{
            'ticker': 'B',
            'quotes': pd.DataFrame(
                {
                    'c': [1, 2, 3, 4, 5, 6],
                    'exec': [2, 3, 4, 5, 6, 7],
                },
                index=[pd.Timestamp(d) for d in ['2018-01-01', '2018-01-02', '2018-01-03',
                                                 '2018-01-07', '2018-01-08', '2018-01-09']]
            ),
            'costs': {
                'type': 'percent',
                'value': 0.5,
            },
            'is_synthetic': True,
        })


    def test__calc_transations_new(self):
        new_pos = {self.asset1: (2, 2, 3)}
        old_pos = None

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(1, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_curr_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]
        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(1, t_position_action)
        self.assertEqual(2, t_curr_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 2, t_costs_close)
        self.assertEqual(-3 * 0.5 * 2, t_costs_exec)

        self.assertEqual(pnl_close_total, 0 + t_costs_close)
        self.assertEqual(pnl_execution_total, 0 + t_costs_exec)
        self.assertEqual(costs_close_total, t_costs_close)
        self.assertEqual(costs_exec_total, t_costs_exec)
        self.assertEqual(costs_potential_close_total, t_costs_close)
        self.assertEqual(costs_potential_exec_total, t_costs_exec)

    def test__calc_transations_close_old(self):
        new_pos = {}
        old_pos = {self.asset1: (2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(1, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_curr_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(-1, t_position_action)
        self.assertEqual(-2, t_curr_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 2, t_costs_close)
        self.assertEqual(-3 * 0.5 * 2, t_costs_exec)

        self.assertEqual(pnl_close_total, 2 + t_costs_close)
        self.assertEqual(pnl_execution_total, 4 + t_costs_exec)
        self.assertEqual(costs_close_total, t_costs_close)
        self.assertEqual(costs_exec_total, t_costs_exec)
        self.assertEqual(costs_potential_close_total, 0.0)
        self.assertEqual(costs_potential_exec_total, 0.0)

    def test__calc_transations_change_existing_decrease(self):
        new_pos = {self.asset1: (1, 2, 3)}
        old_pos = {self.asset1: (2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(1, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(-1, t_position_action)
        self.assertEqual(-1, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 1, t_costs_close)
        self.assertEqual(-3 * 0.5 * 1, t_costs_exec)

        self.assertEqual(pnl_close_total, 2 + t_costs_close)
        self.assertEqual(pnl_execution_total, 4 + t_costs_exec)
        self.assertEqual(costs_close_total, t_costs_close)
        self.assertEqual(costs_exec_total, t_costs_exec)
        self.assertEqual(costs_potential_close_total, -2 * 0.5 * 1)
        self.assertEqual(costs_potential_exec_total, -3 * 0.5 * 1)

    def test__calc_transations_change_existing_reversal_long(self):
        new_pos = {self.asset1: (1, 2, 3)}
        old_pos = {self.asset1: (-2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(2, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(-1, t_position_action)
        self.assertEqual(2, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 2, t_costs_close)
        self.assertEqual(-3 * 0.5 * 2, t_costs_exec)
        self.assertEqual(-2 * 0.5 * 2 - 2, t_pnl_close)
        self.assertEqual(-3 * 0.5 * 2 - 4, t_pnl_exec)

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[1]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(1, t_position_action)
        self.assertEqual(1, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 1, t_costs_close)
        self.assertEqual(-3 * 0.5 * 1, t_costs_exec)
        self.assertEqual(-2 * 0.5 * 1, t_pnl_close)
        self.assertEqual(-3 * 0.5 * 1, t_pnl_exec)

        self.assertEqual(pnl_close_total, -2 + -2*0.5*3)
        self.assertEqual(pnl_execution_total, -4 + -3*0.5*3)
        self.assertEqual(costs_close_total, -2*0.5*3)
        self.assertEqual(costs_exec_total, -3*0.5*3)
        self.assertEqual(costs_potential_close_total, -2 * 0.5 * 1)
        self.assertEqual(costs_potential_exec_total, -3 * 0.5 * 1)

    def test__calc_transations_change_existing_reversal_short(self):
        new_pos = {self.asset1: (-1, 2, 3)}
        old_pos = {self.asset1: (2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(2, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(-1, t_position_action)
        self.assertEqual(-2, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 2, t_costs_close)
        self.assertEqual(-3 * 0.5 * 2, t_costs_exec)

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[1]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(1, t_position_action)
        self.assertEqual(-1, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 1, t_costs_close)
        self.assertEqual(-3 * 0.5 * 1, t_costs_exec)

        self.assertEqual(pnl_close_total, 2 + -2*0.5*3)
        self.assertEqual(pnl_execution_total, 4 + -3*0.5*3)
        self.assertEqual(costs_close_total, -2*0.5*3)
        self.assertEqual(costs_exec_total, -3*0.5*3)
        self.assertEqual(costs_potential_close_total, -2 * 0.5 * 1)
        self.assertEqual(costs_potential_exec_total, -3 * 0.5 * 1)

    def test__calc_transations_change_existing_increase(self):
        new_pos = {self.asset1: (3, 2, 3)}
        old_pos = {self.asset1: (2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(1, len(transactions))

        (
            t_dt,
            t_asset,
            t_position_action,
            t_trans_qty,
            t_close_price,
            t_exec_price,
            t_costs_close,
            t_costs_exec,
            t_pnl_close,
            t_pnl_exec,
        ) = transactions[0]

        self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
        self.assertEqual(self.asset1, t_asset)
        self.assertEqual(1, t_position_action)
        self.assertEqual(1, t_trans_qty)
        self.assertEqual(2, t_close_price)
        self.assertEqual(3, t_exec_price)
        self.assertEqual(-2 * 0.5 * 1, t_costs_close)
        self.assertEqual(-3 * 0.5 * 1, t_costs_exec)

        self.assertEqual(pnl_close_total, 2 + t_costs_close)
        self.assertEqual(pnl_execution_total, 4 + t_costs_exec)
        self.assertEqual(costs_close_total, t_costs_close)
        self.assertEqual(costs_exec_total, t_costs_exec)
        self.assertEqual(costs_potential_close_total, -2 * 0.5 * 3)
        self.assertEqual(costs_potential_exec_total, -3 * 0.5 * 3)

    def test__calc_transations_change_existing_multiple(self):
        new_pos = {self.asset1: (3, 2, 3), self.asset2: (1, 2, 3)}
        old_pos = {self.asset1: (2, 1, 1)}  # Old was opened at '2018-01-01'

        (
            transactions,
            pnl_close_total, pnl_execution_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = Account._calc_transactions(pd.Timestamp('2018-01-02'), new_pos, old_pos)

        self.assertEqual(2, len(transactions))

        for t in transactions:
            (
                t_dt,
                t_asset,
                t_position_action,
                t_trans_qty,
                t_close_price,
                t_exec_price,
                t_costs_close,
                t_costs_exec,
                t_pnl_close,
                t_pnl_exec,
            ) = t
            if t_asset == self.asset1:
                self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
                self.assertEqual(self.asset1, t_asset)
                self.assertEqual(1, t_position_action)
                self.assertEqual(1, t_trans_qty)
                self.assertEqual(2, t_close_price)
                self.assertEqual(3, t_exec_price)
                self.assertEqual(-2 * 0.5 * 1, t_costs_close)
                self.assertEqual(-3 * 0.5 * 1, t_costs_exec)
            elif t_asset == self.asset2:
                self.assertEqual(pd.Timestamp('2018-01-02'), t_dt)
                self.assertEqual(self.asset2, t_asset)
                self.assertEqual(1, t_position_action)
                self.assertEqual(1, t_trans_qty)
                self.assertEqual(2, t_close_price)
                self.assertEqual(3, t_exec_price)
                self.assertEqual(-2 * 0.5 * 1, t_costs_close)
                self.assertEqual(-3 * 0.5 * 1, t_costs_exec)

        self.assertEqual(pnl_close_total, 2 + (-2 * 0.5 * 1) + (-2 * 0.5 * 1) )
        self.assertEqual(pnl_execution_total, 4 + (-3 * 0.5 * 1) + (-3 * 0.5 * 1))
        self.assertEqual(costs_close_total, (-2 * 0.5 * 1) + (-2 * 0.5 * 1))
        self.assertEqual(costs_exec_total, (-3 * 0.5 * 1) + (-3 * 0.5 * 1))
        self.assertEqual(costs_potential_close_total, -2 * 0.5 * 4)
        self.assertEqual(costs_potential_exec_total, -3 * 0.5 * 4)



    def test_init_test(self):
        acc = Account(buffer_len=6, kw=True)

        self.assertEqual({}, acc._position)
        self.assertEqual(0, acc._equity_close)
        self.assertEqual(0, acc._equity_exec)
        self.assertEqual(0, acc._capital_invested)
        self.assertEqual(0, acc._margin)
        self.assertEqual('GenericAccount', acc.name)
        self.assertEqual([], acc._transactions)
        self.assertEqual({'kw': True}, acc.kwargs)
        self.assertEqual(False, acc._has_synthetic_assets)
        self.assertEqual(0, acc._buf_cnt)
        self.assertEqual(6, acc._buffer_len)

        self.assertEqual(np.float, acc._pnl_array_close.dtype)
        self.assertEqual(acc._buffer_len, len(acc._pnl_array_close))

        self.assertEqual(np.float, acc._pnl_array_exec.dtype)
        self.assertEqual(acc._buffer_len, len(acc._pnl_array_exec))

        self.assertEqual(np.float, acc._costs_array_close.dtype)
        self.assertEqual(acc._buffer_len, len(acc._costs_array_close))

        self.assertEqual(np.float, acc._costs_array_exec.dtype)
        self.assertEqual(acc._buffer_len, len(acc._costs_array_exec))

        self.assertEqual(np.float, acc._costs_array_potential_close.dtype)
        self.assertEqual(acc._buffer_len, len(acc._costs_array_potential_close))

        self.assertEqual(np.float, acc._equity_array_close.dtype)
        self.assertEqual(acc._buffer_len, len(acc._equity_array_close))

        self.assertEqual(np.float, acc._equity_array_exec.dtype)
        self.assertEqual(acc._buffer_len, len(acc._equity_array_exec))

        self.assertEqual(np.float, acc._capital_invested_array.dtype)
        self.assertEqual(acc._buffer_len, len(acc._capital_invested_array))

        self.assertEqual(np.float, acc._margin_array.dtype)
        self.assertEqual(acc._buffer_len, len(acc._margin_array))

        self.assertEqual(np.dtype('M8[us]'), acc._date_array.dtype)
        self.assertEqual(acc._buffer_len, len(acc._date_array))

    def test_capital_transaction(self):
        acc = Account(buffer_len=6, kw=True)
        self.assertEqual(0, acc._equity_close)
        self.assertEqual(0, acc._equity_exec)
        self.assertEqual(0, acc._capital_invested)

        acc.capital_transaction(None, 1000)
        self.assertEqual(1000, acc._equity_close)
        self.assertEqual(1000, acc._equity_exec)
        self.assertEqual(1000, acc._capital_invested)

        acc.capital_transaction(None, -1000)
        self.assertEqual(0, acc._equity_close)
        self.assertEqual(0, acc._equity_exec)
        self.assertEqual(0, acc._capital_invested)

    def test_process_position(self):
        with mock.patch('yauber_backtester._account.Account._calc_transactions') as mock_calc_trans:
            with mock.patch('yauber_backtester._account.Account._calc_account_margin') as mock_acc_margin:
                mock_calc_trans.return_value = (
                        ['trans1', 'trans2'],
                        100, 200,
                        -0.5, -1.0,
                        -3, -4,
                )
                mock_acc_margin.return_value = 999

                acc = Account(buffer_len=6, kw=True)
                acc.capital_transaction(pd.Timestamp('2018-01-02'), 1000)

                pos1 = {self.asset1: 1}
                acc._process_position(pd.Timestamp('2018-01-02'), pos1)

                self.assertEqual(False, acc._has_synthetic_assets)
                self.assertEqual(acc._position, {self.asset1: (1, 2, 3)})
                self.assertEqual(True, mock_calc_trans.called)
                self.assertEqual((pd.Timestamp('2018-01-02'), {self.asset1: (1, 2, 3)}, {}), mock_calc_trans.call_args[0])

                self.assertEqual(['trans1', 'trans2'], acc._transactions)
                self.assertEqual(1100, acc._equity_close)
                self.assertEqual(1200, acc._equity_exec)

                self.assertEqual((pd.Timestamp('2018-01-02'), ), mock_acc_margin.call_args[0])
                self.assertEqual(999, acc._margin)

                self.assertEqual(1, acc._buf_cnt)

                self.assertEqual(pd.Timestamp('2018-01-02'), acc._date_array[acc._buf_cnt - 1])
                self.assertEqual(100, acc._pnl_array_close[acc._buf_cnt - 1])
                self.assertEqual(200, acc._pnl_array_exec[acc._buf_cnt - 1])
                self.assertEqual(1100, acc._equity_array_close[acc._buf_cnt - 1])
                self.assertEqual(1200, acc._equity_array_exec[acc._buf_cnt - 1])
                self.assertEqual(1000, acc._capital_invested_array[acc._buf_cnt - 1])
                self.assertEqual(-0.5, acc._costs_array_close[acc._buf_cnt - 1])
                self.assertEqual(-1.0, acc._costs_array_exec[acc._buf_cnt - 1])
                self.assertEqual(-3, acc._costs_array_potential_close[acc._buf_cnt - 1])
                self.assertEqual(-4, acc._costs_array_potential_exec[acc._buf_cnt - 1])
                self.assertEqual(999, acc._margin_array[acc._buf_cnt - 1])

                # Synthetic asset flag setting
                pos1 = {self.asset2: 1}
                acc._process_position(pd.Timestamp('2018-01-02'), pos1)
                self.assertEqual(True, acc._has_synthetic_assets)

                # Check buffer overflow
                acc._buf_cnt = 10
                # ValueError: Incorrectly initialized account values buffer length or _process_position() called more times than expected
                self.assertRaises(ValueError, acc._process_position, pd.Timestamp('2018-01-02'), pos1)

    def test_process_position_errors_checks(self):
        with mock.patch('yauber_backtester._account.Account._calc_transactions') as mock_calc_trans:
            with mock.patch('yauber_backtester._account.Account._calc_account_margin') as mock_acc_margin:
                mock_calc_trans.return_value = (
                        ['trans1', 'trans2'],
                        100, 200,
                        -0.5, -1.0,
                        -3, -4,
                )
                mock_acc_margin.return_value = 999

                acc = Account(buffer_len=6, kw=True)
                acc.capital_transaction(pd.Timestamp('2018-01-02'), 1000)

                self.assertRaises(ValueError, acc._process_position, pd.Timestamp('2018-01-02'), [])
                self.assertRaises(ValueError, acc._process_position, pd.Timestamp('2018-01-02'), {self.asset1: 'se'})
                self.assertRaises(ValueError, acc._process_position, pd.Timestamp('2018-01-02'), {'nope': 1})

    def test_calc_account_margin(self):
        acc = Account(buffer_len=6, kw=True)
        asset1 = mock.MagicMock(Asset)
        asset2 = mock.MagicMock(Asset)

        asset1.get_margin_requirements.return_value = 100
        asset2.get_margin_requirements.return_value = 200

        acc._position = {
            asset1: (1, 2, 3),
            asset2: (2, 2, 3)
        }

        margin = acc._calc_account_margin(pd.Timestamp('2018-01-02'))
        self.assertEqual(margin, 300)
        self.assertEqual((pd.Timestamp('2018-01-02'), 1), asset1.get_margin_requirements.call_args[0])
        self.assertEqual((pd.Timestamp('2018-01-02'), 2), asset2.get_margin_requirements.call_args[0])

    def test_as_asset(self):
        with mock.patch('yauber_backtester._account.Account._calc_transactions') as mock_calc_trans:
            with mock.patch('yauber_backtester._account.Account._calc_account_margin') as mock_acc_margin:
                mock_calc_trans.return_value = (
                        ['trans1', 'trans2'],
                        100, 200,
                        -0.5, -1.0,
                        -3, -4,
                )
                mock_acc_margin.return_value = 999

                acc = Account(buffer_len=6, name='acc')
                acc.capital_transaction(None, 1000)

                pos_dict = {self.asset1: 1}
                acc._process_position(pd.Timestamp('2018-01-02'), pos_dict)

                # Use default acc name
                synt_asset = acc.as_asset()
                self.assertEqual('acc', synt_asset.ticker)

                synt_asset = acc.as_asset("synt1")
                quotes = synt_asset.quotes()


                self.assertEqual('synt1', synt_asset.ticker)
                self.assertEqual(1, len(quotes))
                self.assertEqual(1100, quotes['o'][0])
                self.assertEqual(1100, quotes['h'][0])
                self.assertEqual(1100, quotes['l'][0])
                self.assertEqual(1100, quotes['c'][0])
                self.assertEqual(0, quotes['v'][0])
                self.assertEqual(1200, quotes['exec'][0])

                self.assertEqual(True, synt_asset.is_synthetic)
                self.assertEqual(1.0, synt_asset.get_point_value(pd.Timestamp('2018-01-02')))

                self.assertEqual(1, len(synt_asset.kwargs['margin']))
                self.assertEqual(999, synt_asset.kwargs['margin'][0])
                self.assertEqual(999, synt_asset.get_margin_requirements(pd.Timestamp('2018-01-02'), 1))
                self.assertEqual({'A': 1}, synt_asset.legs)

                self.assertEqual('dynamic', synt_asset.kwargs['costs']['type'])
                self.assertEqual(1, len(synt_asset.kwargs['costs']['value']))
                self.assertEqual(-3, synt_asset.kwargs['costs']['value']['c'][0])
                self.assertEqual(-4, synt_asset.kwargs['costs']['value']['exec'][0])

                #
                # Disallow creating synth asset from accounts holding another synth asset
                #
                acc = Account(buffer_len=6)
                pos_dict = {self.asset2: 1}
                acc._process_position(pd.Timestamp('2018-01-02'), pos_dict)
                # ValueError: It's not permitted to create multiple layers of synthetic assets. This account already contains one or more synthetic assets.
                self.assertRaises(ValueError, acc.as_asset, "synt1")


    def test_position(self):
        acc = Account(buffer_len=6, kw=True)
        asset1 = mock.MagicMock(Asset)
        asset2 = mock.MagicMock(Asset)


        acc._position = {
            asset1: (1, 2, 3),
            asset2: (2, 2, 3)
        }

        p = acc.position()

        self.assertEqual(True, isinstance(p, dict))

        self.assertEqual(True, asset1 in p)
        self.assertEqual(True, asset2 in p)

        self.assertEqual(asset1, p[asset1].asset)
        self.assertEqual(asset2, p[asset2].asset)
        self.assertEqual(1, p[asset1].qty)
        self.assertEqual(2, p[asset2].qty)

    def test_public_properties(self):
        acc = Account(buffer_len=6, name='test')
        acc._equity_close = 100
        acc._capital_invested = 200
        acc._margin = 25

        self.assertEqual(100, acc.capital_equity)
        self.assertEqual(200, acc.capital_invested)
        self.assertEqual(25, acc.margin)
        self.assertEqual(75, acc.capital_available)
        self.assertEqual('test', acc.name)

    def test_builtins(self):
        acc = Account(buffer_len=6, name='test')
        acc2 = Account(buffer_len=6, name='test')
        acc3 = Account(buffer_len=6, name='test3')
        self.assertEqual(str(acc), 'test')
        self.assertEqual(hash(acc), hash('test'))
        self.assertEqual(True, acc == acc2)
        self.assertEqual(True, acc != acc3)
        self.assertEqual(True, acc == 'test')
        self.assertEqual(False, acc == None)

    def test_as_transactions(self):
        acc = Account(buffer_len=6, name='test')
        acc._transactions = [
            # 'date', 'asset', 'position_action', 'qty', 'price_close', 'price_exec',
            #                                 'costs_close', 'costs_exec', 'pnl_close', 'pnl_execution',
            (
                pd.Timestamp('2018-01-01'),
                self.asset1,
                1,
                1,
                100,
                101,
                -0.5,
                -0.6,
                3,
                4,
            ),
            (
                pd.Timestamp('2018-01-02'),
                self.asset2,
                1,
                -1,
                100,
                101,
                -0.5,
                -0.6,
                3,
                4,
            ),
        ]

        df = acc.as_transactions()

        self.assertEqual(True, isinstance(df, pd.DataFrame))
        self.assertEqual(True, df.index.is_monotonic_increasing)
        self.assertEqual(2, len(df))
        self.assertEqual([
                                'asset', 'position_action', 'qty', 'price_close', 'price_exec',
                                'costs_close', 'costs_exec', 'pnl_close', 'pnl_execution',
                         ],
                         list(df.columns),
        )
        self.assertEqual(df.index.name, 'date')

    def test_as_dataframe(self):
        with mock.patch('yauber_backtester._account.Account._calc_transactions') as mock_calc_trans:
            with mock.patch('yauber_backtester._account.Account._calc_account_margin') as mock_acc_margin:
                mock_calc_trans.return_value = (
                        ['trans1', 'trans2'],
                        100, 200,
                        -0.5, -1.0,
                        -3, -4,
                )
                mock_acc_margin.return_value = 999

                acc = Account(buffer_len=6, name='acc')
                acc.capital_transaction(None, 1000)

                pos_dict = {self.asset1: 1}
                acc._process_position(pd.Timestamp('2018-01-02'), pos_dict)

                df = acc.as_dataframe()

                self.assertEqual(1, len(df))
                self.assertEqual(5, len(df.columns))

                self.assertEqual(1200, df['equity'][0])
                self.assertEqual(1000, df['capital_invested'][0])
                self.assertEqual(-1.0, df['costs'][0])
                self.assertEqual(999, df['margin'][0])
                self.assertEqual(200, df['pnl'][0])
                self.assertEqual(pd.Timestamp('2018-01-02'), df.index[0])


if __name__ == '__main__':
    unittest.main()

