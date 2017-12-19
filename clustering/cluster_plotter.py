import numpy as np
import matplotlib.pyplot as pyplot
from itertools import cycle

def plot_cluster(X,labels,cluster_centers=None,title="Clustered Data"):
	"""Plots 2-D datapoint with a labelwise colors.

	Parameters
	----------
	X : array-like of shape (n_samples, 2)
		2-D points to be plotted. 
	labels : array-like of shape (n_samples,)
		Provides label for each datatpoint
	cluster_centers : array-like of shape (n_samples,), optional
		Array of clusters centers. 
		If provided Line conection from point to center will be established.
	title : strint, optional
		Plot title. Default is "Clustered Data".
	"""
	pyplot.figure()
	pyplot.clf()
	X = np.array(X)
	labels = np.array(labels)
	if cluster_centers is not None:
		cluster_centers = np.array(cluster_centers)

	colors = cycle('bgrcmyk')

	for i, (l,col) in enumerate(zip(np.unique(labels),colors)):
		class_members = labels == l
		pyplot.scatter(X[class_members, 0], X[class_members, 1], s=10, color=col)
		
		if cluster_centers is not None:
			pyplot.plot(cluster_centers[i,0], cluster_centers[i,1], 'o', markerfacecolor=col,
			markeredgecolor='k', markersize=14)
			for x in X[class_members]:
				pyplot.plot([cluster_centers[i,0], x[0]], [cluster_centers[i,1], x[1]], col)

	pyplot.title(title)
	#pyplot.show()