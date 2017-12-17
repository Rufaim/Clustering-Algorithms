import numpy as np


def euclidean_distances(X,squared=True):
	X = np.atleast_2d(X)
	distance = np.sum((X[np.newaxis,:,:]-X[:,np.newaxis,:])**2,axis=2)
	distance = np.maximum(0, distance)
	distance.flat[::distance.shape[0] + 1] = 0.0
	return distance if squared else np.sqrt(distance)