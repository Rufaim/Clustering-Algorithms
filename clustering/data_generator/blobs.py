import numpy as np


def make_blobs(n_samples=100, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, seed=None):
    """Generate Gaussian blobs for clustering. Each sample is two dimentional for easy plotting.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points equally divided among clusters.
    centers : int or array of shape [n_centers, 2], optional(default=3)
        The number of centers to generate, or the fixed center locations.
    cluster_std : float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
    center_box : pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : boolean, optional (default=True)
        Shuffle the samples.
    seed : int or None, optional (default=None)
        seed used by the random number generator;
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.
    """
    np.random.seed(seed)

    if type(centers) is int:
        centers = np.random.uniform(center_box[0], center_box[1],size=(centers, 2))
    else:
        centers = np.array(centers)
        assert centers.ndim == 2
        assert centers.shape[1] == 2

    if type(cluster_std) in [int,float]:
        cluster_std = float(cluster_std)
        cluster_std = np.ones(len(centers)) * cluster_std

    X,y = [],[]

    n_centers = centers.shape[0]
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers

    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + np.random.normal(scale=std,size=(n, 2)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y
