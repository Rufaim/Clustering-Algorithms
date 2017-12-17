import numpy as np
import enum
from .interfaces import ClusterAlghorihm
from .utility import euclidean_distances


def affinity_propagation(S, convergence_iter=15, max_iter=100, discount=0.5):
	N = S.shape[0]

	A = np.zeros((N, N))
	R = np.zeros((N, N))

	t = np.zeros((N, N))
	index = np.arange(N)

	e = np.zeros((N, convergence_iter))

	for it in range(max_iter):
		#section_1 r(i,k) = s(i,k) - max{k' != k} ( a(i,k') + s(i,k'))

		t = A+S
		max_elem_idx = np.argmax(t, axis=1)
		max_elem = t[index, max_elem_idx]
		t[index, max_elem_idx] = -np.inf
		second_max_elem = np.max(t, axis=1)
		t = S - max_elem[:, None]
		t[index, max_elem_idx] = S[index, max_elem_idx] - second_max_elem
		#section_1_end

		# exponential update
		R = discount * R + (1 - discount) * t
		
		#section_2 {i != k} a(i,k) = min(0, r(k,k) + sum{j != i, j != k } (max(0,r(j,k))))
		#					a(k,k) = sum{j != k}(max(0,r(j,k)))
		t = np.maximum(0,R)
		t.flat[::N + 1] = R.flat[::N + 1]
		t = np.sum(t, axis=0)-t
		diag_elems = t.flat[::N + 1]
		t = np.minimum(0,t)
		t.flat[::N + 1] = diag_elems
		#section_2_end

		A = discount * A + (1 - discount) * t

		#section_3 convergence check

		E = (np.diag(A) + np.diag(R)) > 0
		e[:, it % convergence_iter] = E
		K = np.sum(E, axis=0)

		if it >= convergence_iter:
			se = np.sum(e, axis=1)
			unconverged = (np.sum((se == convergence_iter) + (se == 0))!= N)
			if (not unconverged and (K > 0)) or (it == max_iter):
				break
		#section_3_end

	I = np.where(E)[0] # I = np.where(np.diag(A + R) > 0)[0]
	K = len(I)  # Identify exemplars

	if K > 0:
		C = np.argmax(A+R, axis=1)
		cluster_centers_idx = np.unique(C)
		labels = np.searchsorted(cluster_centers_idx, C)
	else:
		# unconvergened case
		labels = np.empty((N, 1))
		cluster_centers_idx = None
		labels.fill(np.nan)

	return cluster_centers_idx, labels



class AffinityPropagation(ClusterAlghorihm):
	"""Affinity Propagation Clustering alghorithm.

	Parameters
	----------
	discount_factor : float, optional, default: 0.5
		Discount_factor for exponetial update. Between 0 and 1.
	max_iter : int, optional, default: 200
		Maximum number of iterations.
	convergence_iter : int, optional, default: 15
		Number of iterations with no change in the number
		of clusters that marks the convergence.
	preference : array-like of shape (n_samples,) or float, optional, default: None
	    Self-affinity of point. Points with larger values are more likely
	    to be chosen as cluster center. Defaultly set as median of fit input.
	affinity : AffinityType, optional, default : EUCLIDIAN
		Type of affinity to be used.
		If PRECOMPUTED, then affinity matrix meant to be passed to fit.
		If EUCLIDIAN, then negative squared euclidean distance will be used.

    Attributes
    ----------
	cluster_centers_idx : array of shape (n_clusters,)
		Indices of cluster centers
    labels : array, shape (n_samples,)
        Cluster labels of each point.

	References
	----------
	Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
	Between Data Points", Science Feb. 2007
	"""

	class AffinityType(enum.Enum):
		"""Types of affinity will be passed into fit method
		"""
		EUCLIDIAN = 0
		PRECOMPUTED = 1

	def __init__(self, discount_factor=.5, max_iter=200, convergence_iter=15,
				preference=None, affinity=AffinityType.EUCLIDIAN):
		assert max_iter > 0
		assert convergence_iter > 0

		if not isinstance(affinity,AffinityPropagation.AffinityType):
			raise ValueError('affinity should be of affinity_type')
		if discount_factor <= 0 or discount_factor >= 1:
			raise ValueError('Discount factor should be 0 < DF < 1')

		self.discount_factor = discount_factor
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.preference = preference
		self.affinity = affinity

	def fit(self,X):
		X = np.atleast_2d(X)

		assert X.ndim == 2

		if self.affinity is self.AffinityType.PRECOMPUTED:
			S = X.copy()

			if S.shape[0] != S.shape[1]:
				raise ValueError("Precomputed affinity should be a square array")
		elif self.affinity ==  self.AffinityType.EUCLIDIAN:
			S = -euclidean_distances(X, squared=True)

		if self.preference is None:
			preference = np.median(S)
		S.flat[::(S.shape[0] + 1)] = preference

		self.cluster_centers_idx, self.labels  = affinity_propagation(S,
				convergence_iter=self.convergence_iter, max_iter=self.max_iter, discount=self.discount_factor)

		return self

