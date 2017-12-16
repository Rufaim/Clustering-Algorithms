import numpy as np

def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=100, discount=0.5, seed=None):
	S = np.array(S, copy=True)
	
	assert S.ndim == 2

	if S.shape[0] != S.shape[1]:
		raise ValueError("S should be a square array")

	if preference is None:
		preference = np.median(S)
	if discount <= 0 or discount >= 1:
		raise ValueError('Discount factor should be 0 < DF < 1')

	N = S.shape[0]

	np.random.seed(seed)


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
		t[index, max_elem_idx] = S[:, max_elem_idx] - second_max_elem
		#section_1_end

		# exponential update
		R = discount * R + (1 - discount) * t
		
		#section_2 {i != k} a(i,k) = min(0, r(k,k) + sum{j != i, j != k } (max(0,r(j,k))))
		#					a(k,k) = sum{j != k}(max(0,r(j,k)))
		t = np.maximum(0,R)
		t.flat[::N + 1] = R.flat[::N + 1]
		t = np.sum(t, axis=0)[:,None]-t
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
