import numpy as np
from collections import Iterable

def make_circles(n_samples=100, centers=3, center_box=(-10.0, 10.0),
                radiuses_box=(1.0,5.0),radiuses=None, noise_std=None, shuffle=True,seed=None):
    """Make a large circle containing a smaller circle in 2d.
    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points generated.
    centers : int or array of shape [n_centers, 2], optional(default=3)
        The number of centers to generate, or the fixed center locations.
    noise_std : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    radiuses_box : pair of floats (min, max) or array of shape [n_centers,], optional (default=(1.0, 5.0))
        The bounds for each cluster radius.
    radiuses : array of shape [n_centers,], optional (default=None)
        The array of radiuses for each cluser. If None provided, will be generated randomly.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    seed : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    np.random.seed(seed)

    if type(centers) is int:
        centers = np.random.uniform(center_box[0], center_box[1],size=(centers, 2))
    else:
        centers = np.array(centers)
        assert centers.ndim == 2
        assert centers.shape[1] == 2

    n_centers = centers.shape[0]
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers

    if radiuses is not None:
        rads = np.array(radiuses)
    else:
        rads = np.random.uniform(radiuses_box[0], radiuses_box[1],size=(n_centers, ))    
    

    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    linspace = np.linspace(0, 2 * np.pi, int(n_samples // n_centers) ,endpoint=False).reshape((-1,1))
    circ = np.hstack([np.cos(linspace),np.sin(linspace)])
    
    X,y = [],[]   

    for i, n in enumerate(n_samples_per_center):
        X.append(centers[i] + rads[i]*circ)
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)
    if shuffle:
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    if noise_std is not None:
        X += np.random.normal(scale=noise_std, size=X.shape)

    return X, y