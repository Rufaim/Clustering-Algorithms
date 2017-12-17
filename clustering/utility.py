import numpy as np


def euclidean_distances(X,Y=None,squared=True):
	"""Euclidian distatnces between two vectors
	
	Parameters
	----------
	X : array-like of shape (n_samples_1, n_features)
	Y : array-like of shape (n_samples_2, n_features)
	squared : bool, optional
		Returns squared Euclidean distances if true.

	Returns
	-------
	distances : ndarray of shape (n_samples_1, n_samples_2)

	"""
	X = np.atleast_2d(X)
	if Y is None:
		Y = X.copy()
	else:
		assert Y.shape[1] == X.shape[1]
	distance = np.sum((X[:,np.newaxis,:]-Y[np.newaxis,:])**2,axis=2)
	distance = np.maximum(0, distance)
	if Y is None:
		# make sure distance to self is zero 
		distance.flat[::distance.shape[0] + 1] = 0.0
	return distance if squared else np.sqrt(distance)