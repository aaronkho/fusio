import copy
import warnings
import logging
import numpy as np
from numpy.typing import ArrayLike, NDArray

from scipy.integrate import cumulative_simpson  # type: ignore[import-untyped]

logger = logging.getLogger('fusio')


def vectorized_numpy_derivative(
    x: NDArray,
    y: NDArray,
) -> NDArray:
    deriv = np.zeros_like(x)
    if x.shape[-1] > 2:
        x1 = np.concatenate([np.expand_dims(x[..., 0], axis=-1), x[..., :-2], np.expand_dims(x[..., -3], axis=-1)], axis=-1)
        x2 = np.concatenate([np.expand_dims(x[..., 1], axis=-1), x[..., 1:-1], np.expand_dims(x[..., -2], axis=-1)], axis=-1)
        x3 = np.concatenate([np.expand_dims(x[..., 2], axis=-1), x[..., 2:], np.expand_dims(x[..., -1], axis=-1)], axis=-1)
        y1 = np.concatenate([np.expand_dims(y[..., 0], axis=-1), y[..., :-2], np.expand_dims(y[..., -3], axis=-1)], axis=-1)
        y2 = np.concatenate([np.expand_dims(y[..., 1], axis=-1), y[..., 1:-1], np.expand_dims(y[..., -2], axis=-1)], axis=-1)
        y3 = np.concatenate([np.expand_dims(y[..., 2], axis=-1), y[..., 2:], np.expand_dims(y[..., -1], axis=-1)], axis=-1)
        deriv = ((x - x1) + (x - x2)) / (x3 - x1) / (x3 - x2) * y3 + ((x - x1) + (x - x3)) / (x2 - x1) / (x2 - x3) * y2 + ((x - x2) + (x - x3)) / (x1 - x2) / (x1 - x3) * y1
    elif x.shape[-1] > 1:
        deriv[..., 0] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
        deriv[..., 1] = np.diff(y, axis=-1) / np.diff(x, axis=-1)
    return deriv

def vectorized_numpy_integration(
    y: NDArray,
    x: NDArray,
) -> NDArray:
    return cumulative_simpson(y, x=x, initial=0.0)

def vectorized_numpy_interpolation(
    v: float | NDArray,
    x: NDArray,
    y: NDArray,
    extrapolate: bool = False,
) -> NDArray:
    vm = np.array([v]) if isinstance(v, float) else copy.deepcopy(v)
    xm = x.reshape(-1, x.shape[-1])
    ym = y.reshape(-1, y.shape[-1])
    if vm.shape[0] != xm.shape[0]:
        vm = np.repeat(np.expand_dims(vm, axis=0), xm.shape[0], axis=0)
    interp = np.zeros_like(vm)
    for i in range(xm.shape[0]):
        interp[i] = np.interp(vm[i], xm[i], ym[i])
        um = vm[i] > np.nanmax(xm[i])
        if np.any(um) and extrapolate:
            s = (ym[i, -1] - ym[i, -2]) / (xm[i, -1] - xm[i, -2])
            interp[i][um] = ym[i, -1] + s * (vm[i][um] - xm[i, -1])
        lm = vm[i] < np.nanmin(xm[i])
        if np.any(lm) and extrapolate:
            s = (ym[i, 0] - ym[i, 1]) / (xm[i, 0] - xm[i, 1])
            interp[i][um] = ym[i, 0] + s * (vm[i][um] - xm[i, 0])
    interp = interp.reshape(*y.shape[:-1])
    return interp

def vectorized_numpy_find(
    v: float | NDArray,
    x: NDArray,
    y: NDArray,
    last: bool = False,
) -> NDArray:
    xm = x.reshape(-1, x.shape[-1])
    ym = y.reshape(-1, y.shape[-1])
    flat_found = np.full((xm.shape[0], ), np.nan)
    for i in range(xm.shape[0]):
        yidx = np.where(((ym[i] - v)[1:] * (ym[i] - v)[:-1]) < 0.0)[0]
        if len(yidx) > 0:
            yi = yidx[-1] if last else yidx[0]
            flat_found[i] = (v - ym[i, yi]) * (xm[i, yi + 1] - xm[i, yi]) / (ym[i, yi + 1] - ym[i, yi])
    found = flat_found.reshape(*y.shape[:-1])
    return found