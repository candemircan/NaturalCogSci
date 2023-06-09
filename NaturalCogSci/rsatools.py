# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/rsatools.ipynb.

# %% auto 0
__all__ = ['cka']

# %% ../nbs/rsatools.ipynb 2
# | code-fold: false
import numpy as np

from .helpers import get_project_root

# %% ../nbs/rsatools.ipynb 3
def cka(
    X: np.ndarray,  # Representations of the first set of samples.
    Y: np.ndarray,  # Representations of the second set of samples.
) -> float:  # The linear CKA between X and Y.
    """
    Compute the linear CKA between two matrices X and Y.

    [link to the paper](https://arxiv.org/abs/1905.00414)

    taken from Patrick Mineault's implementation of CKA as is.

    [link to original implementation](https://goodresearch.dev/cka.html)

    Matrices should be observations by features.

    """
    # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX**2).sum() / np.sqrt((XTX**2).sum() * (YTY**2).sum())
