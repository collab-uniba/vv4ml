"""Latin Hypercube Sampling"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_chunked as pdc

from vv4ml.config import InputField


def _lh_basic(
    par_num: int, sample_size: int, rand_gen: np.random.Generator
) -> NDArray[np.float64]:
    """Base Latin hypercube sampling.

    Generate samples by splitting the [0,1] interval into `sample_size` segments and
    map sample coordinates into them. Finally, permutate every dimension to guarantee
    the Latin hypercube property.

    Parameters
    ----------
    par_num: int
        Numbers of parameters to sample (dimension space).
    sample_size: int
        Number of sample points to be generated in the space.
    rand_gen: NumPy random number generator instance.

    Returns
    -------
    sam_coord: NumPy float 2D array (`sample_size` rows, `par_num` columns).
        Each row stores the `par_num` coordinates of a sample point.
    """
    lh_lim = np.linspace(0, 1, sample_size + 1)[np.newaxis].T
    sam_coord = rand_gen.random((sample_size, par_num))
    sam_coord = sam_coord * (lh_lim[1:] - lh_lim[:-1]) + lh_lim[:-1]
    for i in range(par_num):
        sam_coord[:, i] = sam_coord[rand_gen.permutation(sample_size), i]

    return sam_coord


def _lh_maximin(
    par_num: int, sample_size: int, rand_gen: np.random.Generator, iter_num: int
) -> NDArray[np.float64]:
    """Maximin criterion applied to the base Latin hypercube sampling.

    Every `iter_num` iteration, generate a base lh sample and compute Euclidean distances of
    every chunk and find the optimal candidate by searching for the smallest distance.

    Parameters
    ----------
    par_num: int
        Numbers of parameters to sample (dimension space).
    sample_size: int
        Number of sample points to be generated in the space.
    rand_gen: NumPy random number generator instance.
    iter_num: int
        Number of tries where base lhs candidates are generated.

    Returns
    -------
    optim_lh: NumPy float 2D array (`sample_size` rows, `par_num` columns).
        Best sample coordinates found where points have the biggest minimum distance.
    """
    lh_min_dist = 0
    for _ in range(iter_num):
        cand_dist = np.inf
        cand_coord = _lh_basic(par_num, sample_size, rand_gen)

        dist_iter = pdc(cand_coord, metric="euclidean")
        for mat_chunk in dist_iter:
            min_dist = np.min(mat_chunk, where=(mat_chunk != 0), initial=np.inf)
            if cand_dist > min_dist:
                cand_dist = min_dist

        if lh_min_dist < cand_dist:
            lh_min_dist = cand_dist
            optim_lh = cand_coord

    return optim_lh


def lh_gen(
    par_num: int,
    sample_size: int,
    optimization: Optional[Literal["maximin"]] = None,
    iter_num: int = 5,
    rand_gen: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.float64]:
    """Generate a Latin Hypercube Sampling (LHS) design.

    Parameters
    ----------
    par_num : int
        Number of parameters to sample (dimension space).
    sample_size : int
        Number of sample points to be generated in the space.
    optimization : str, optional
        Method to choose the best sample candidate. Currently, the only supported
        method is `maximin` (see the note below).
    iter_num : int, default 5
        Number of iterations to find the best sampling. Unused if `optimization`
        is `None`.
    rand_gen : NumPy random number generator instance, optional.
        Optional parameter to pass a custom random number generator.

    Returns
    -------
    samples : NumPy float 2D array (`sampleSize` rows, `parNum` columns)
        Each row stores the `parNum` coordinates of a sample point.

    Notes
    -----
    The optimization method `maximin` choose the best sampling as that where the
    minimum Euclidean distance between each couple of sample points is the largest.
    Due to the very large number of comparisons, it is discouraged to use this method
    for large numbers (>1000) of sample points.
    """
    if optimization is None:
        samples = _lh_basic(par_num, sample_size, rand_gen)
    elif optimization == "maximin":
        samples = _lh_maximin(par_num, sample_size, rand_gen, iter_num)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization!r}")

    return samples


def generate_lhs_samples(
    inputs: list[InputField],
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate LHS samples scaled to the configured input ranges.

    Parameters
    ----------
    inputs : list[InputField]
        Input field specifications from the configuration.
    sample_size : int
        Number of sample points to generate.
    random_state : int, default 42
        Seed for the random number generator.

    Returns
    -------
    Pandas DataFrame with `sample_size` rows and one column per input field,
        values scaled to each field's configured range.
    """
    rand_gen = np.random.default_rng(random_state)
    samples = lh_gen(par_num=len(inputs), sample_size=sample_size, rand_gen=rand_gen)

    # Scale each column from [0, 1] to the input field's range
    for i, field in enumerate(inputs):
        lo, hi = field.range
        samples[:, i] = lo + samples[:, i] * (hi - lo)

    column_names = [field.name for field in inputs]
    return pd.DataFrame(samples, columns=column_names)
