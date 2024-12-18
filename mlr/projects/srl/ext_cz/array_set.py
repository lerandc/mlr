from __future__ import annotations
import numpy as np
from czone.util.eset import EqualSet
from collections.abc import Iterable



class ArraySet(EqualSet):
    """Equality based set class. Checks strict array equality."""

    def __init__(self, x: np.ndarray | list[np.ndarray] = None):
        """
        Args:
            x: np.ndarray or list of np.ndarrays, arrays which initialize set. 
               If sole array is provided, will separate sub arrays along first dimension, 
               i.e., as through the default numpy iteration order.
        """
        super().__init__(x)

    @staticmethod
    def _equality_check(x, y):
        return np.array_equal(x, y)
    
    def __contains__(self, item):
        for x in self:
            if self._equality_check(x, item):
                return True
        return False
    
    def remove(self, other: Iterable[np.ndarray]):
        tmp = self._make_unique(other)
        for t in tmp:
            for i, x in enumerate(self):
                if self._equality_check(x, t):
                    del self._storage[i]
                    break

class NormedArraySet(ArraySet):
    """Equality based set class. Checks array equality based on normed difference, with tolerance."""

    def __init__(self, x: np.ndarray | list[np.ndarray], tol: float, norm_ord=None):
        """
        Args:
            x: np.ndarray or list of np.ndarrays, arrays which initialize set.
               If sole array is provided, will separate sub arrays along first dimension, 
               i.e., as through the default numpy iteration order.
            tol: float, tolerance level for equality check.
            norm_ord: norm order, passed to np.linalg.norm
        """
        self.norm_ord = norm_ord
        self.tol = tol
        super().__init__(x)

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, val: float):
        if float(val) >= 0.0:
            self._tol = val
        else:
            raise ValueError(f"Tolerance must be nonnegative. Received: {val:0.4f}")
        
    def to_set(self, other):
        return self.__class__(other, tol=self.tol, norm_ord=self.norm_ord)

    def _equality_check(self, x, y):
        return np.linalg.norm((x - y), ord=self.norm_ord) < self.tol
    