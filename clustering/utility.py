import numpy as np


def euclidean_distances(X,Y=None,squared=True):
	X = np.atleast_2d(X)
	if Y is None:
		Y = X.copy()
	else:
		assert Y.shape[0] == X.shape[0]
	distance = np.sum((X[:,np.newaxis,:]-Y[np.newaxis,:])**2,axis=2)
	distance = np.maximum(0, distance)
	if Y is None:
		# make sure distance to self is zero 
		distance.flat[::distance.shape[0] + 1] = 0.0
	return distance if squared else np.sqrt(distance)