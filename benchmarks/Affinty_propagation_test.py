import numpy as np
from clustering import datasets, plot_cluster, AffinityPropagation
import matplotlib.pyplot as pyplot

def test(dataset,preference,discount_factor=0.5):
	X,y,title = dataset()
	plot_cluster(X,y,title=title)

	ap = AffinityPropagation(preference=preference,max_iter=200,convergence_iter=30,discount_factor=discount_factor)
	ap.fit(X)

	plot_cluster(X,ap.labels,X[ap.cluster_centers_idx],title=title+ " clustered")

def test1():
	test(datasets.load_WingNut,-2)

def test2():
	test(datasets.load_compound,-50)

def test3():
	test(datasets.load_pathbased,-500)

def test4():
	test(datasets.load_spiral,-70)

def test5():
	test(datasets.load_R15,-10)

def test6():
	test(datasets.load_Jain_toy_dataset,-300)

def test7():
	test(datasets.load_flame,-80)

def test8():
	test(datasets.load_Target,-50,.95)

def test9():
	test(datasets.load_EngyTime,-1000,.95)

def test10():
	test(datasets.load_TwoDiamonds,-100,.95)

def test11():
	test(datasets.load_D31,-100,.95)



if __name__ == '__main__':
	test1()
	test2()
	test3()
	test4()
	test5()
	test6()
	test7()
	test8()
	test9()
	test10()
	test11()
	pyplot.show()