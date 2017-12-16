import numpy as np
import matplotlib.pyplot as pyplot
from itertools import cycle

def plot_cluster(X,labels,title="Clusered Data"):
	pyplot.figure()
	pyplot.clf()

	colors = cycle('bgrcmyk')
	for l,col in zip(np.unique(labels),colors):
		class_members = labels == l
		pyplot.scatter(X[class_members, 0], X[class_members, 1], s=10, color=col)

	pyplot.title(title)
	pyplot.show()


def plot_centroidal_clusters(X,cluster_centers,labels,title="Clusered Data"):
	pyplot.figure()
	pyplot.clf()

	colors = cycle('bgrcmyk')
	cluster_centers = np.array(cluster_centers)
	for k, col in zip(range(len(cluster_centers)), colors):
		class_members = labels == k
		pyplot.plot(X[class_members, 0], X[class_members, 1], col + '.')
		pyplot.plot(cluster_centers[k,0], cluster_centers[k,1], 'o', markerfacecolor=col,
			markeredgecolor='k', markersize=14)
		for x in X[class_members]:
			pyplot.plot([cluster_centers[k,0], x[0]], [cluster_centers[k,1], x[1]], col)

	pyplot.title(title)
	pyplot.show()
