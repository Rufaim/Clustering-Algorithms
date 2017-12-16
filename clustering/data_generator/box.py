import numpy as np

def make_box(n_samples=100, clusters=3, upper_boundaries=(5.0, 5.0),
             shuffle=True, seed=None):
    """Generate Gaussian blobs for clustering. Each sample is two dimentional for easy plotting.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The total number of points equally divided among clusters.
    clusters : int, optional(default=3)
        The number of clusters to generate.
    upper_boundaries : pair of floats (x_man, y_max), optional (default=(5.0, 5.0))
        The bounds for box dimentions for being generated at random.
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
    upper_boundaries = np.array(upper_boundaries)
    box_corner = np.random.uniform(upper_boundaries/3, upper_boundaries,size=(clusters, 2))

    X,y = [],[]
    n_samples_per_center = [int(n_samples // clusters)] * clusters

    for i in range(n_samples % clusters):
        n_samples_per_center[i] += 1

    for i, n in enumerate(n_samples_per_center):
        X.append(np.random.uniform([0,0], box_corner[i],size=(n, 2)) + np.random.uniform(-2*upper_boundaries,2*upper_boundaries,size=(1, 2)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y