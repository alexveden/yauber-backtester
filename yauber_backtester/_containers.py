from ._asset import Asset
from collections import namedtuple, OrderedDict
import numpy as np
from numpy import take as np_take
from numpy import argsort as np_argsort
from typing import Tuple

import numba
from math import nan


@numba.jit(nopython=True)
def _unstack(values, n_assets, n_fields):  # pragma: no cover
    result = np.full((n_assets, n_fields), nan)

    for i in range(n_assets):
        for j in range(n_fields):
            result[i, j] = values[i * n_fields + j]
    return result


class PositionInfo:
    """
    Container for position information
    """
    __slots__ = ['asset', 'qty']  # Decrease memory footprint

    def __init__(self, asset, qty):
        self.asset: Asset = asset
        """Asset of the opened position"""

        self.qty: float = qty
        """Quantity of the opened position"""

    def __str__(self):
        return f"{self.asset} x {self.qty}"

    def __repr__(self):
        return self.__str__()


class RowTuple:
    """
    Fast named key-value row wrapper around np.ndarray
    """
    __slots__ = ['names', 'values']

    def __init__(self, names):
        self.names = {c: i for i, c in enumerate(names)}
        self.values = np.full(len(names), nan)

    def fill(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[self.names[key]]


class MFrame:
    """
    Strategy metric frame
    """
    def __init__(self, assets, columns):
        self.shape = (len(assets), len(columns))
        self._data = np.full(self.shape, nan)
        self._columns = OrderedDict([(c, i) for i, c in enumerate(columns)])
        self._columns_list = tuple(columns)
        self._assets = OrderedDict([(a, i) for i, a in enumerate(assets)])
        self._assets_list = np.array(assets)
        self._assets_tickers = OrderedDict([(a.ticker, a) for a in assets])
        self._row_tuple = RowTuple(self._columns.keys())
        self._indexes = np.array(range(len(assets)))

    def _fill(self, metric_matrix):
        _shape = self.shape
        _unst = _unstack(metric_matrix, _shape[0], _shape[1])
        assert _shape == _unst.shape
        self._data = _unst

    def items(self) -> Tuple[Asset, RowTuple]:
        """
        Iterate over items of MFrame
        :return: iterator of tuple (Asset, RowTuple)
        """
        _data = self._data
        r = self._row_tuple
        for a, i in self._assets.items():
            r.fill(_data[i])
            yield a, r

    @property
    def columns(self):
        """
        List of available asset metrics / columns
        :return:
        """
        return self._columns_list

    @property
    def assets(self):
        """
        List of available assets
        :return:
        """
        return self._assets_list

    def __getitem__(self, key) -> np.ndarray:
        """
        Return metric array across all assets
        :param key: column name
        :return: np.ndarray length of assets
        """
        return self._data[:, self._columns[key]]

    def get_at(self, asset, metric) -> float:
        """
        Return scalar value of metric for asset
        :param asset: Asset class instance or ticker string
        :param metric: column name
        :return: float
        """
        return self._data[self._assets[asset], self._columns[metric]]

    def get_asset(self, asset_ticker) -> Asset:
        """
        Get asset by ticker
        :param asset_ticker: ticker string
        :return:
        """
        return self._assets_tickers[asset_ticker]

    def get_filtered(self, condition, sort_by_col=None):
        """
        Filter and optionally sort asset metrics by condition
        :param condition: boolean array (example: (mf['some_metric'] > 0) & (mf['another_metric'] == 1) )
        :param sort_by_col: (optional) column name to sort results (sort order is always ascending)
        :return: tuple of arrays ( sorted_assets_array, sorted_metrics_data_matrix)
        """
        flt_idx = self._indexes[condition]
        _flt_data = self._data.take(flt_idx, axis=0)
        if sort_by_col is not None:
            col_id = self._columns[sort_by_col]

            srt_idx = np_argsort(_flt_data[:, col_id])
            orig_sorted_idx = np_take(flt_idx, srt_idx, axis=0)
            return np_take(self._assets_list, orig_sorted_idx), np_take(_flt_data, srt_idx, axis=0)
        else:
            return np_take(self._assets_list, flt_idx), self._data[flt_idx, :]
