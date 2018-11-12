from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from math import isfinite


class Asset:
    """
    Generic asset class
    """

    def __init__(self, **kwargs):
        self.ticker = kwargs['ticker']
        self._quotes = kwargs['quotes']

        if not isinstance(self._quotes, pd.DataFrame):
            raise ValueError(f'Asset "{self}" quotes type must be Pandas.DataFrame, got type <{type(self._quotes)}>')
        else:
            if len(self._quotes) == 0:
                raise ValueError(f"Empty quotes series for {self}")
            if 'c' not in self._quotes or 'exec' not in self._quotes:
                raise ValueError(f'Asset "{self}" quotes dataframe must contain at least "c" and "exec" columns and , got {self._quotes.columns}')
        #
        # Setting quotes cache for fast access
        #
        self._quotes_values = self._quotes.values
        self._quotes_col_close = self._quotes.columns.get_loc('c')
        self._quotes_col_exec = self._quotes.columns.get_loc('exec')

        self.kwargs = kwargs

        #
        #   Setting costs functions (to speed up code at execution time)
        #
        self._costs_func = self._costs_func_zero
        self._costs_value = None

        if 'costs' in self.kwargs:
            costs_dict = self.kwargs['costs']
            if not isinstance(costs_dict, dict) or 'type' not in costs_dict or 'value' not in costs_dict:
                raise ValueError("'costs' in asset's kwargs must be a dict with {'type': ... and 'value': ... } keys")

            if costs_dict['type'] == 'percent':
                if not isinstance(costs_dict['value'], (float, np.float)):
                    raise ValueError("'costs' value of 'percent' type must be a single float number")
                if costs_dict['value'] < 0:
                    raise ValueError("'costs' value of 'percent' type must be positive")
                self._costs_value = costs_dict['value']
                self._costs_func = self._costs_func_percent

            elif costs_dict['type'] == 'dollar':
                if not isinstance(costs_dict['value'], (float, np.float, int, np.int32, np.int64)):
                    raise ValueError("'costs' value of 'dollar' type must be a single float number")
                if costs_dict['value'] < 0:
                    raise ValueError("'costs' value of 'dollar' type must be positive")
                self._costs_value = costs_dict['value']
                self._costs_func = self._costs_func_dollar

            elif costs_dict['type'] == 'dynamic':
                if not isinstance(costs_dict['value'], pd.DataFrame) or 'c' not in costs_dict['value'] or 'exec' not in costs_dict['value']:
                    raise ValueError("'costs' value of 'dynamic' type must be a Pandas.DataFrame with columns ['c', 'exec']")
                if not self._quotes.index.equals(costs_dict['value'].index):
                    raise ValueError("'costs' value of 'dynamic' dataframe must have the same length and index as quotes")
                self._costs_value = costs_dict['value']
                self._costs_func = self._costs_func_dynamic
            else:
                raise ValueError(f"Unknown costs type {costs_dict['type']}, only 'percent', 'dollar', 'dynamic' are supported")

        #
        # Asset margin requirements
        #
        self.margin = self.kwargs.get('margin', None)
        if self.margin is not None:
            if isinstance(self.margin, pd.Series):
                # We have dynamic margin requirements for the asset
                if not self._quotes.index.equals(self.margin.index):
                    raise ValueError("'margin' pd.Series must have the same length and index as quotes")
            elif isinstance(self.margin,  (float, np.float, int, np.int32, np.int64)):
                if self.margin < 0:
                    raise ValueError("'margin' must be >= 0")
            else:
                raise ValueError("'margin' unsupported type of asset margin, it must be pd.Series or float")

        #
        # Support of multileg assets
        #
        self.legs = self.kwargs.get('legs', None)
        """Most recent composition of multileg asset. This method might be used to get live position composition."""

        if self.legs is None:
            self.legs = {self.ticker: 1.0}
        else:
            if not isinstance(self.legs, dict):
                raise ValueError("Asset 'legs' must be a dictionary of {<ticker_string>: <qty_float>}")
            for k, v in self.legs.items():
                if not isinstance(k, str):
                    raise ValueError(f"Asset 'legs' keys must be strings, got {type(k)}")
                if not isinstance(v, (float, np.float, int, np.int32, np.int64)):
                    raise ValueError(f"Asset 'legs' values must be numbers, got {type(v)}")

        #
        # Asset point value
        #
        self._point_value = self.kwargs.get('point_value', 1.0)
        if isinstance(self._point_value, pd.Series):
            # We have dynamic point value for the asset
            if not self._quotes.index.equals(self._point_value.index):
                raise ValueError("'point_value' pd.Series must have the same length and index as quotes")
        elif isinstance(self._point_value, (float, np.float, int, np.int32, np.int64)):
            if self._point_value <= 0:
                raise ValueError("'pointvalue' must be > 0")
        else:
            raise ValueError(f"'point_value' unsupported type, it must be pd.Series or float, got {type(self._point_value)}")

        #
        # Caching
        #
        self._cache_px_date = None
        self._cache_px_result = None
        self._cache_pointvalue_date = None
        self._cache_pointvalue_result = None

    def __hash__(self):
        return hash(self.ticker)

    def __str__(self):
        return self.ticker

    def __repr__(self):
        return f'{self.__class__.__name__}<{self.ticker}>'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ticker == other

        if not isinstance(other, Asset):
            return False

        return self.ticker == other.ticker

    @property
    def is_synthetic(self):
        return self.kwargs.get('is_synthetic', False)

    def quotes(self, **kwargs):
        """
        Get asset quotes dataframe
        """
        return self._quotes

    def get_prices(self, date) -> Tuple[float, float]:
        """
        Get Close and Execution price at 'date'
        :param date:
        :return: tuple (close px, exec px)
        """
        if self._cache_px_date == date:
            # Use cached prices if we had previous request at the same date
            return self._cache_px_result
        try:
            # Try fast way
            idx = self._quotes.index.get_loc(date)
            result = self._quotes_values[idx][self._quotes_col_close], self._quotes_values[idx][self._quotes_col_exec]
        except KeyError:
            ser = self._quotes.loc[:date]
            if len(ser) == 0:
                raise KeyError(f'No quotes found at {date}, quotes range {self._quotes.index[0]} - {self._quotes.index[-1]}')

            result = ser['c'][-1], ser['exec'][-1]

        self._cache_px_date = date
        self._cache_px_result = result
        return result

    def get_point_value(self, date) -> float:
        """
        Return dollar value per 1 point (execution time)
        :param date:
        :return:
        """
        if self._cache_pointvalue_date == date:
            return self._cache_pointvalue_result

        if isinstance(self._point_value, pd.Series):
            try:
                # Try fast way
                result = self._point_value.at[date]
            except KeyError:
                ser = self._point_value.loc[:date]
                if len(ser) == 0:
                    raise KeyError(f'No point value found at {date}, range {self._point_value.index[0]} - {self._point_value.index[-1]}')
                result = ser[-1]

            if result <= 0:
                raise ValueError(f'Point value for the asset {self} is <= 0 at {date} value: {result}')

            result = result
            self._cache_pointvalue_date = date
            self._cache_pointvalue_result = result
            return result
        else:
            self._cache_pointvalue_date = date
            self._cache_pointvalue_result = self._point_value
            return self._point_value

    def get_costs(self, date, qty) -> Tuple[float, float]:
        """
        Calculate asset's transaction costs in dollars at specific date

        :param date: calculation date
        :param qty: Transaction quantity
        :return: tuple (close time costs, exec time costs px)
        """
        # self._costs_func is dynamically defined based on costs settings see. __init__()
        return self._costs_func(date, qty)

    def _costs_func_zero(self, date, qty):
        return 0.0, 0.0

    def _costs_func_percent(self, date, qty):
        cpx, epx = self.get_prices(date)
        return -abs(cpx * self._costs_value * qty), -abs(epx * self._costs_value * qty)

    def _costs_func_dollar(self, date, qty):
        return -abs(self._costs_value * qty), -abs(self._costs_value * qty)

    def _costs_func_dynamic(self, date, qty):
        try:
            # Try fast way
            ccosts, ecosts = self._costs_value.at[date, 'c'], self._costs_value.at[date, 'exec']
        except KeyError:
            ser = self._costs_value.loc[:date]
            if len(ser) == 0:
                raise KeyError(f'No costs found at {date}, costs range {self._costs_value.index[0]} - {self._costs_value.index[-1]}')
            ccosts, ecosts = ser['c'][-1], ser['exec'][-1]

        return -abs(ccosts * qty), -abs(ecosts * qty)

    def get_margin_requirements(self, date, qty) -> float:
        """
        Get margin requirements for the asset (at execution time)
        :param date: calculation date
        :param qty: quantity of opened position
        :return:
        """
        if self.margin is None:
            # if no margin settings, use cash-like margin (100% margin requirements)
            return self.calc_position_value(date, qty)
        else:
            if isinstance(self.margin, pd.Series):
                try:
                    # Try fast way
                    result = self.margin.at[date]
                except KeyError:
                    ser = self.margin.loc[:date]
                    if len(ser) == 0:
                        raise KeyError(f'No margin found at {date}, margin range {self.margin.index[0]} - {self.margin.index[-1]}')
                    result = ser[-1]

                if result < 0:
                    raise ValueError(f'Margin requirements for the asset {self} is negative at {date} value: {result}')
                return result * abs(qty)
            else:
                if self.margin <= 1.0:
                    # Margin is floating number < 1.0
                    # Use percent margin in this case
                    pval = self.calc_position_value(date, qty)
                    return pval * self.margin
                else:
                    # Margin is floating number > 1.0
                    # Use absolute dollar margin in this case
                    return self.margin * abs(qty)

    def calc_position_value(self, date, qty) -> float:
        """
        Calculate dollar position value at the specific 'date' (execution time)
        :param date:
        :return:
        """
        cpx, epx = self.get_prices(date)
        if isfinite(epx):
            _valid_price = epx
        elif isfinite(cpx):
            _valid_price = cpx
        else:
            raise ValueError(f"Invalid asset price for {self} at {date}")

        return _valid_price * self.get_point_value(date) * abs(qty)

    def calc_dollar_pnl(self, date, prev_price, current_price, qty) -> float:
        """
        Calculate dollar PnL for asset

        :param date: calculation date
        :param prev_price: previous price
        :param current_price: current price
        :param qty:
        :return: PnL in dollars
        """
        return (current_price - prev_price) * qty * self.get_point_value(date)
