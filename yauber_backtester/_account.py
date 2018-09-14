from typing import Dict
from ._asset import Asset
from datetime import datetime
import pandas as pd
import numpy as np
from ._containers import PositionInfo





class Account:
    """
    Generic position management class
    """
    def __init__(self, buffer_len, **kwargs):
        """
        Initialize backtester account
        :param buffer_len: set the buffer length according to underlying quotes length
        :param kwargs:
        """
        self.kwargs = kwargs

        self.name = kwargs.get('name', 'GenericAccount')

        self._position = {}
        self._transactions = []

        self._equity_close = 0.0
        self._equity_exec = 0.0
        self._capital_invested = 0.0
        self._margin = 0.0
        self._has_synthetic_assets = False
        self._buf_cnt = 0
        self._buffer_len = buffer_len

        self._date_array = np.full(self._buffer_len, np.datetime64('NaT'), dtype=np.dtype('M8[us]'))
        self._pnl_array_close = np.full(self._buffer_len, np.nan)
        self._pnl_array_exec = np.full(self._buffer_len, np.nan)
        self._costs_array_close = np.full(self._buffer_len, np.nan)
        self._costs_array_exec = np.full(self._buffer_len, np.nan)
        self._costs_array_potential_close = np.full(self._buffer_len, np.nan)
        self._costs_array_potential_exec = np.full(self._buffer_len, np.nan)
        self._equity_array_close = np.full(self._buffer_len, np.nan)
        self._equity_array_exec = np.full(self._buffer_len, np.nan)
        self._capital_invested_array = np.full(self._buffer_len, np.nan)
        self._margin_array = np.full(self._buffer_len, np.nan)

        self.capital_transaction(None, kwargs.get('initial_capital', 0))



    @property
    def capital_equity(self):
        """
        Most recent equity value (at close price)
        :return:
        """
        return self._equity_close

    @property
    def capital_invested(self):
        """
        Net capital invested by account.capital_transaction()
        :return:
        """
        return self._capital_invested

    @property
    def capital_available(self):
        """
        Amount of capital available for opening new positions: account.capital_equity - account.margin
        :return:
        """
        return self._equity_close - self._margin

    @property
    def margin(self):
        """
        Total account position margin requirement
        :return:
        """
        return self._margin

    def position(self) -> Dict[Asset, PositionInfo]:
        """
        Returns information about opened position
        :return: dictionary {asset: PositionInfoClass, ...}
        """
        result = {}

        for asset, v in self._position.items():
            result[asset] = PositionInfo(asset, v[0])

        return result

    def capital_transaction(self, dt, amount, is_own_money=True):
        """
        Add or withdraw capital
        :param amount:
        :param is_own_money: if True - deposit/withdraw own money, if False - it might be interest, or dividends
        :return:
        """
        self._equity_close += amount
        self._equity_exec += amount
        if is_own_money:
            self._capital_invested += amount

    def as_transactions(self) -> pd.DataFrame:
        """
        Returns list of account transactions as Pandas.DataFrame, with datetime index and columns:
        'asset', 'position_action', 'qty', 'price_close', 'price_exec', 'costs_close', 'costs_exec', 'pnl_close', 'pnl_execution'
        :return:
        """
        df = pd.DataFrame(self._transactions,
                            columns=[
                                'date', 'asset', 'position_action', 'qty', 'price_close', 'price_exec',
                                'costs_close', 'costs_exec', 'pnl_close', 'pnl_execution',
                            ]).set_index('date')

        assert df.index.is_monotonic_increasing
        return df

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return dataframe of account's arrays of :
        - 'equity' (at exec time)
        - 'capital_invested'
        - 'costs' (at exec time)
        - 'margin'
        - 'pnl' (at exec time)
        :return:
        """
        return pd.DataFrame(
                {
                    'equity': self._equity_array_exec[:self._buf_cnt],
                    'capital_invested': self._capital_invested_array[:self._buf_cnt],
                    'costs': self._costs_array_exec[:self._buf_cnt],
                    'margin': self._margin_array[:self._buf_cnt],
                    'pnl': self._pnl_array_exec[:self._buf_cnt],
                },
                index=self._date_array[:self._buf_cnt],
            )

    def as_asset(self, name=None) -> Asset:
        """
        Creates synthetic asset based on account's equity and costs
        :return:
        """
        if self._has_synthetic_assets:
            raise ValueError("It's not permitted to create multiple layers of synthetic assets. "
                             "This account already contains one or more synthetic assets.")

        if name is None:
            name = self.name

        # Replace negative equity margin by zeros
        _margin = np.where(self._margin_array[:self._buf_cnt] < 0, 0, self._margin_array[:self._buf_cnt])

        synt_asset = {
            'ticker': name,
            'quotes': pd.DataFrame(
                {
                    'o': self._equity_array_close[:self._buf_cnt],
                    'h': self._equity_array_close[:self._buf_cnt],
                    'l': self._equity_array_close[:self._buf_cnt],
                    'c': self._equity_array_close[:self._buf_cnt],
                    'v': np.zeros(self._buf_cnt),
                    'exec': self._equity_array_exec[:self._buf_cnt],
                },
                index=self._date_array[:self._buf_cnt],
            ),
            'is_synthetic': True,  # To forbid creating multiple asset derivatives from accounts
            'point_value': 1.0,    # Equity prices are in dollars, so point value should be always 1.0
            'margin': pd.Series(_margin, index=self._date_array[:self._buf_cnt]),
            'legs': {k.ticker: v[0] for k, v in self._position.items()},  # 'legs' must be a dictionary of {<ticker_string>: <qty_float>}
            'costs': {
                'type': 'dynamic',
                'value': pd.DataFrame({
                    # IMPORTANT: we use only potential costs because the transaction costs are included in equity prices
                    'c': self._costs_array_potential_close[:self._buf_cnt],
                    'exec': self._costs_array_potential_exec[:self._buf_cnt],
                }, index=self._date_array[:self._buf_cnt]),
            }
        }
        return Asset(**synt_asset)

    #
    #
    # Protected methods
    #       (!!!) DO NOT USE in strategy analysis
    #
    #
    def _process_position(self, dt: datetime, new_pos: Dict[Asset, float]):
        """
        Processed new position calculates transactions and PnLs
        :param dt:
        :param new_pos:
        :return:
        """
        # 1. Convert new_pos to dictionary of {asset: Tuple (PosQuantity, ClosePrice, ExecPrice), ... }
        if not isinstance(new_pos, dict):
            raise ValueError(f'strategy.compose_portfolio() must return dict of <asset_AssetClassInstance: qty_FloatNumber>, got <{type(new_pos)}>')

        new_pos_dict = {}
        for asset, qty in new_pos.items():
            if not isinstance(asset, Asset) or not isinstance(qty, (float, np.float, int, np.int32, np.int64)):
                raise ValueError(f'strategy.compose_portfolio() must return dict of <asset_AssetClassInstance: qty_FloatNumber>,'
                                 f' got <{type(asset)}: {type(qty)}>')

            self._has_synthetic_assets = self._has_synthetic_assets or asset.is_synthetic
            close_price, exec_price = asset.get_prices(dt)
            new_pos_dict[asset] = (qty, close_price, exec_price)

        # Calculate transactions logic for positions
        (
            transactions,
            pnl_close_total, pnl_exec_total,
            costs_close_total, costs_exec_total,
            costs_potential_close_total, costs_potential_exec_total,
        ) = self._calc_transactions(dt, new_pos_dict, self._position)

        # Update position PnL values
        self._transactions += transactions
        self._equity_close += pnl_close_total
        self._equity_exec += pnl_exec_total
        self._position = new_pos_dict
        self._margin = self._calc_account_margin(dt)

        # Build historical arrays
        i = self._buf_cnt
        if i >= self._buffer_len:
            raise ValueError("Incorrectly initialized account values buffer length or _process_position() called more times than expected")

        self._date_array[i] = dt
        self._pnl_array_close[i] = pnl_close_total
        self._pnl_array_exec[i] = pnl_exec_total
        self._equity_array_close[i] = self._equity_close
        self._equity_array_exec[i] = self._equity_exec
        self._capital_invested_array[i] = self._capital_invested

        self._costs_array_close[i] = costs_close_total
        self._costs_array_exec[i] = costs_exec_total
        self._costs_array_potential_close[i] = costs_potential_close_total
        self._costs_array_potential_exec[i] = costs_potential_exec_total
        self._margin_array[i] = self._margin
        self._buf_cnt += 1

    def _calc_account_margin(self, dt):
        """
        Calculates summary account margin
        :return:
        """
        margin = 0.0

        for asset, (qty, cpx, epx) in self._position.items():
            margin += asset.get_margin_requirements(dt, qty)

        return margin

    @staticmethod
    def _calc_transactions(dt, current_position_dict, prev_position_dict):
        """
        Should return
        - daily pnl (at close and exec price)
        - costs realized (at close and exec price)
        - costs potential (if we decide to manage opened position somewhere else)
        - list of transactions
        :param dt:
        :param current_position_dict: dict of {<asset>: (<qty>, <close px>, <exec px>)}
        :param prev_position_dict: dict of {<asset>: (<qty>, <close px>, <exec px>)}
        :return:
        """
        pnl_close_total = 0.0
        pnl_execution_total = 0.0
        costs_close_total = 0.0
        costs_exec_total = 0.0
        costs_potential_close_total = 0.0
        costs_potential_exec_total = 0.0
        transactions = []

        assert current_position_dict is not None, 'current_pos must be initialized'

        if prev_position_dict is None:
            intersected_assets = set(current_position_dict.keys())
        else:
            intersected_assets = set(current_position_dict.keys()) | set(prev_position_dict.keys())

        for asset in intersected_assets:
            # prev_pos = Tuple (PosQuantity, ClosePrice, ExecPrice)
            prev_pos = prev_position_dict.get(asset, None) if prev_position_dict is not None else None
            # curr_pos = Tuple (PosQuantity, ClosePrice, ExecPrice)
            curr_pos = current_position_dict.get(asset, None)

            if curr_pos is not None:
                curr_qty, close_price, exec_price = curr_pos
                costs_close, costs_exec = asset.get_costs(dt, curr_qty)
                costs_potential_close_total += costs_close
                costs_potential_exec_total += costs_exec

            if prev_pos is None:
                if curr_pos[0] != 0:
                    #
                    # Open new position
                    #
                    pnl_close = 0.0 + costs_close
                    pnl_execution = 0.0 + costs_exec
                    position_action = 1  # 1 - open new position, -1 - close old position, 0 - hold position

                    # Store stats
                    transactions.append((
                        dt,
                        asset,
                        position_action,
                        curr_qty,
                        close_price,
                        exec_price,
                        costs_close,
                        costs_exec,
                        pnl_close,
                        pnl_execution
                    ))
                    pnl_close_total += pnl_close
                    pnl_execution_total += pnl_execution
                    costs_close_total += costs_close
                    costs_exec_total += costs_exec

            elif curr_pos is None:
                if prev_pos[0] != 0:
                    #
                    # Close old position or skip old closed positions
                    #
                    prev_qty, prev_cpx, prev_epx = prev_pos

                    # Costs and prices
                    costs_close, costs_exec = asset.get_costs(dt, -prev_qty)
                    close_price, exec_price = asset.get_prices(dt)

                    pnl_close = asset.calc_dollar_pnl(dt, prev_cpx, close_price, prev_qty) + costs_close
                    pnl_execution = asset.calc_dollar_pnl(dt, prev_epx, exec_price, prev_qty) + costs_exec
                    position_action = -1  # 1 - open new position, -1 - close old position, 0 - hold position

                    # Store stats
                    transactions.append((
                        dt,
                        asset,
                        position_action,
                        -prev_qty,
                        close_price,
                        exec_price,
                        costs_close,
                        costs_exec,
                        pnl_close,
                        pnl_execution
                    ))

                    pnl_close_total += pnl_close
                    pnl_execution_total += pnl_execution
                    costs_close_total += costs_close
                    costs_exec_total += costs_exec
            else:
                # Calculating transactions for existing position
                prev_qty, prev_cpx, prev_epx = prev_pos
                curr_qty, curr_cpx, curr_epx = curr_pos
                trans_qty = curr_qty - prev_qty
                new_trans_qty = 0
                if trans_qty != 0:
                    # Handle reversal transactions
                    if (curr_qty > 0 and prev_qty < 0) or (curr_qty < 0 and prev_qty > 0):
                        trans_qty = -prev_qty
                        new_trans_qty = curr_qty
                else:
                    if curr_qty == 0 and prev_qty == 0:
                        continue

                costs_close, costs_exec = asset.get_costs(dt, trans_qty)
                pnl_close = asset.calc_dollar_pnl(dt, prev_cpx, curr_cpx, prev_qty) + costs_close
                pnl_execution = asset.calc_dollar_pnl(dt, prev_epx, curr_epx, prev_qty) + costs_exec

                position_action = 0  # 1 - open new position, -1 - close old position, 0 - hold position
                abs_pos_chg = abs(curr_qty) - abs(prev_qty)
                if abs_pos_chg > 0:
                    position_action = 1
                elif abs_pos_chg < 0:
                    position_action = -1

                transactions.append((
                    dt,
                    asset,
                    position_action,
                    trans_qty,
                    close_price,
                    exec_price,
                    costs_close,
                    costs_exec,
                    pnl_close,
                    pnl_execution
                ))

                # Update total stats
                pnl_close_total += pnl_close
                pnl_execution_total += pnl_execution
                costs_close_total += costs_close
                costs_exec_total += costs_exec

                if new_trans_qty != 0:
                    #
                    # Handle reversal transactions (to support correct trades list in reports)
                    # If previous size is 2 (long) and current size is -1 (short), we will have to sell -3 but in 2 transactions
                    # sell -2 - to close previous long
                    # sell -1 - to open new short
                    costs_close, costs_exec = asset.get_costs(dt, new_trans_qty)
                    pnl_close = costs_close
                    pnl_execution = costs_exec

                    transactions.append((
                        dt,
                        asset,
                        1,  # position_action, - new position
                        new_trans_qty,
                        close_price,
                        exec_price,
                        costs_close,
                        costs_exec,
                        pnl_close,
                        pnl_execution
                    ))
                    # Update total stats
                    pnl_close_total += pnl_close
                    pnl_execution_total += pnl_execution
                    costs_close_total += costs_close
                    costs_exec_total += costs_exec


        # Return results as tuple because it's faster than dict or Pandas objects
        return (
                    transactions,
                    pnl_close_total, pnl_execution_total,
                    costs_close_total, costs_exec_total,
                    costs_potential_close_total, costs_potential_exec_total,
        )

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if not isinstance(other, Account):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
