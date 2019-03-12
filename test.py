# !usr/bin/env python
# -*- coding:utf-8 -*-
# Author:LiuQian,time:2018/6/7

import numpy as np


def f(x):
    return x**2


g = lambda x: x**2


def mi(x, y, k=3, base=2):
    """ Mutual information of x and y
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * np.rand(len(x[0]))) for p in x]
    y = [list(p + intens * np.rand(len(y[0]))) for p in y]
    points = zip2(x, y)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
    a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    return (-a - b + c + d) / log(base)