import unittest
from yauber_backtester._report import Trade, Report
import pandas as pd
import numpy as np
from unittest import mock
from yauber_backtester._account import Account


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            (
                pd.Timestamp('2017-01-01'),
                'test', # asset
                1,  # position_action,
                1,  #  trans_qty,
                3,  # close_price,
                2,  # exec_price,
                0,  # costs_close,
                0,  # costs_exec,
                1,  # pnl_close,
                2,  #pnl_execution
            ),
            (
                pd.Timestamp('2017-01-01'),
                'test2',  # asset
                1,  # position_action,
                1,  # trans_qty,
                3,  # close_price,
                2,  # exec_price,
                0,  # costs_close,
                0,  # costs_exec,
                1,  # pnl_close,
                2,  # pnl_execution
            ),
            (
                pd.Timestamp('2017-01-02'),
                'test',  # asset
                -1,  # position_action,
                -1,  # trans_qty,
                3,  # close_price,
                2,  # exec_price,
                0,  # costs_close,
                0,  # costs_exec,
                1,  # pnl_close,
                2,  # pnl_execution
            ),
            (
                pd.Timestamp('2017-01-02'),
                'test2',  # asset
                0,  # position_action,
                0,  # trans_qty,
                3,  # close_price,
                2,  # exec_price,
                0,  # costs_close,
                0,  # costs_exec,
                1,  # pnl_close,
                2,  # pnl_execution
            ),

        ]

    def test__produce_trades_list(self):
        acc = Account(buffer_len=5)
        acc._transactions = self.transactions
        df_trades = Report._produce_trades_list(acc)

        self.assertEqual(2, len(df_trades))
        self.assertEqual(True, isinstance(df_trades, pd.DataFrame))



if __name__ == '__main__':
    unittest.main()
