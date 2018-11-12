from typing import Iterable, Tuple
from ._account import Account
import pandas as pd
import numpy as np
from math import isfinite
from collections import OrderedDict


TRADE_KEYS = ('asset', 'date_entry', 'date_exit', 'side', 'n_transactions', 'wavg_price_entered', 'wavg_price_exited',
              'qty_entered', 'qty_exited', 'pnl', 'pnl_perc', 'costs', 'context')
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
        self._qty = transaction['qty']
        self._context = np.nan

        if 'context' in transaction:
            if transaction['context'] is not None:
                self._context = transaction['context']

    @property
    def is_closed(self):
        return self._is_closed

    def as_tuple(self):
        entry_avg_px = self._entry_value / self._entry_qty if self._entry_qty > 0 else np.nan
        exit_avg_px = self._exit_value / self._exit_qty if self._exit_qty > 0 else np.nan
        pnl_perc = (exit_avg_px / entry_avg_px - 1) * self._side
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
            pnl_perc,  # % trade pnl
            self._costs,
            self._context,
        )

    def add_transaction(self, dt, transaction):
        qty = transaction['qty']
        pnl = transaction['pnl_execution']
        costs = transaction['costs_exec']
        exec_px = transaction['price_exec']
        assert transaction['asset'] == self._asset
        assert not ((self._qty > 0 and self._qty + qty < 0) or
                    (self._qty < 0 and self._qty + qty > 0)), f'Reversal transaction detected! {self._asset} at {dt}: ' \
                                                              f'Opened: {self._qty} Trans: {qty}'
        assert not self._is_closed, 'Position already closed'

        if isfinite(pnl):
            self._pnl += pnl
        else:
            if isfinite(transaction['pnl_close']):
                self._pnl += transaction['pnl_close']
                exec_px = transaction['price_close']

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
            self._entry_value += exec_px * abs(qty)
            self._n_transations += 1

        if transaction['position_action'] == -1:
            # Add qty to existing position
            self._exit_qty += abs(transaction['qty'])
            self._exit_value += exec_px * abs(qty)
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
        trade_pnl_perc = acc_trades['pnl_perc'].fillna(0)

        mdd = np.nan
        mdd_pct = np.nan
        winrate = np.nan
        netprofit = np.nan
        netprofit_perc = np.nan
        cagr = np.nan
        trade_avg = np.nan
        trade_std = np.nan

        if len(trade_pnl) > 0 and len(acc_df) > 0:
            equity = acc_df['equity'].ffill()
            capital_invested = acc_df['capital_invested'].ffill()
            capital_invested_avg = capital_invested.mean()
            capital_invested_avg = np.nan if capital_invested_avg == 0 else capital_invested_avg
            pnl_series_perc = acc_df['pnl'].fillna(0) / equity
            equity_perc = equity / capital_invested * 100

            winrate = len(trade_pnl[trade_pnl > 0]) / len(trade_pnl)
            netprofit = equity[-1] - capital_invested_avg
            netprofit_perc = (netprofit / capital_invested_avg)
            mdd_arr = (equity - equity.expanding().max())
            mdd = mdd_arr.min()
            mdd_pct_arr = (equity / equity.expanding().max() - 1)
            mdd_pct = mdd_pct_arr.min()

            difference_in_years = (equity.index[-1] - equity.index[0]).days / 365.2425
            if equity[-1] > 0:
                cagr = (equity[-1] / equity[0]) ** (1 / difference_in_years) - 1
            else:
                cagr = -1.0

            trade_avg = trade_pnl_perc.mean()
            trade_std = trade_pnl_perc.std()

            # Replace account series by %
            acc_df['equity'] = equity_perc
            acc_df['pnl'] = pnl_series_perc
            acc_df['mdd'] = mdd_pct_arr * 100


        stats = pd.Series(OrderedDict([
            ('CAGR %', cagr*100),
            ('NetProfit $', netprofit),
            ('NetProfit %', netprofit_perc * 100),
            ('MaxDD $', mdd),
            ('MaxDD %', mdd_pct * 100),
            ('NumberOfTrades', len(acc_trades)),
            ('WinRate', f'{winrate * 100:0.2f}%'),
            ('Trade % Mean', trade_avg * 100),
            ('Trade % StDev', trade_std * 100),
            ]))

        return stats, acc_df, acc_trades

