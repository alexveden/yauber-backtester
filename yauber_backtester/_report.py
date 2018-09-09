from typing import Iterable, Tuple
from ._account import Account
import pandas as pd
import numpy as np
from math import isfinite

TRADE_KEYS = ('asset', 'date_entry', 'date_exit', 'side', 'n_transactions', 'wavg_price_entered', 'wavg_price_exited',
              'qty_entered', 'qty_exited', 'pnl', 'costs')
"""Trade records static keys for export"""


class Trade:
    def __init__(self, dt, transaction):
        # Expected Transaction keys
        # 'asset', 'position_action', 'qty', 'price_close', 'price_exec', 'costs_close', 'costs_exec', 'pnl_close', 'pnl_execution'
        assert transaction['position_action'] == 1, 'Must be opening transaction'

        self._pnl = transaction['pnl_execution']
        self._n_transations = 1
        self._entry_qty = abs(transaction['qty'])
        self._entry_value = transaction['price_exec'] * self._entry_qty

        self._exit_qty = 0
        self._exit_value = 0

        self._costs = transaction['costs_exec']
        self._entry_date = dt
        self._exit_date = dt
        self._side = 1 if transaction['qty'] > 0 else -1
        self._is_closed = False
        self._asset = transaction['asset']
        self._qty = self._entry_qty

    @property
    def is_closed(self):
        return self._is_closed

    def as_tuple(self):
        entry_avg_px = self._entry_value / self._entry_qty if self._entry_qty > 0 else np.nan
        exit_avg_px = self._exit_value / self._exit_qty if self._exit_qty > 0 else np.nan

        return (
            self._asset,
            self._entry_date,
            self._exit_date,
            self._side,
            self._n_transations,
            entry_avg_px,
            exit_avg_px,
            self._entry_qty,
            self._exit_qty,
            self._pnl,
            self._costs
        )

    def add_transaction(self, dt, transaction):
        qty = transaction['qty']
        pnl = transaction['pnl_execution']
        costs = transaction['costs_exec']
        assert not ((self._qty > 0 and self._qty + qty < 0) or (self._qty < 0 and self._qty + qty > 0)), 'Reversal transaction detected!'
        assert not self._is_closed, 'Position already closed'
        assert transaction['asset'] == self._asset

        if isfinite(pnl):
            self._pnl += pnl
        else:
            if isfinite(transaction['pnl_close']):
                self._pnl += transaction['pnl_close']
        if isfinite(costs):
            self._costs += costs
        else:
            if isfinite(transaction['costs_close']):
                self._costs += transaction['costs_close']
        self._qty += qty
        self._exit_date = dt

        if transaction['position_action'] == 1:
            # Add qty to existing position
            self._entry_qty += abs(qty)
            self._entry_value += transaction['price_exec'] * abs(qty)
            self._n_transations += 1

        if transaction['position_action'] == -1:
            # Add qty to existing position
            self._exit_qty += abs(transaction['qty'])
            self._exit_value += transaction['price_exec'] * abs(qty)
            self._n_transations += 1

            if self._qty == 0:
                self._is_closed = True
                self._exit_date = dt


class Report:
    """
    Generic backtester report
    """
    def __init__(self, accounts: Iterable[Account], **kwargs):
        """
        Build backtester report after initialization
        :param accounts: list of accounts
        """
        self.accounts = accounts
        self.results = {}

        for acc in accounts:
            if acc in self.results:
                raise ValueError(f"Duplicated account name '{acc}'")

            self.results[acc] = self._build(acc)

    def stats(self) -> pd.DataFrame:
        """
        Re
        :return:
        """
        return pd.DataFrame({acc: r[0] for acc, r in self.results.items()})

    def series(self, series_name) -> pd.DataFrame:
        """
        Return dataframe of multiple account series
        :param series_name: (see. Account.as_dataframe)
        Return dataframe of account's arrays of :
        - 'equity' (at exec time)
        - 'capital_invested'
        - 'costs' (at exec time)
        - 'margin'
        - 'pnl' (at exec time)
        :return:
        """
        return pd.DataFrame({acc: r[1][series_name] for acc, r in self.results.items()})

    def trades(self, acc_name) -> pd.DataFrame:
        """
        Returns trades list for specific account name
        :param acc_name: account name
        :return:
        """
        return self.results[acc_name][2]


    @staticmethod
    def _produce_trades_list(account) -> pd.DataFrame:
        """
        Produces trades list using account transactions
        :param account:
        :return:
        """
        all_transactions = account.as_transactions()

        closed_trades = []
        trades = {}

        for dt, trans in all_transactions.iterrows():
            a = trans['asset']
            if a not in trades:
                trades[a] = Trade(dt, trans)
            else:
                t = trades[a]
                t.add_transaction(dt, trans)
                if t.is_closed:
                    closed_trades.append(t)
                    del trades[a]

        # Add all remaining opened trades
        for t in trades.values():
            closed_trades.append(t)
        trade_tuples = [t.as_tuple() for t in closed_trades]
        return pd.DataFrame(trade_tuples, columns=TRADE_KEYS)

    def _build(self, account) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
        """
        Builds backtesting report statistics
        :return:
        """
        acc_df = account.as_dataframe()
        acc_trades = Report._produce_trades_list(account)

        # Calculate stats
        trade_pnl = acc_trades['pnl'].fillna(0)
        equity = acc_df['equity'].ffill()
        mdd = np.nan
        winrate = np.nan
        netprofit = np.nan

        if len(trade_pnl) > 0 and len(acc_df) > 0:
            winrate = len(trade_pnl[trade_pnl > 0]) / len(trade_pnl)
            netprofit = equity[-1]
            mdd = (equity - equity.expanding().max()).min()

        stats = {
            'NumberOfTrades': len(acc_trades),
            'WinRate': winrate,
            'NetProfit': netprofit,
            'MaxDD': mdd
        }

        return stats, acc_df, acc_trades

