import numpy as np
import os

def conventional_reader(path_to_file):
	data = np.genfromtxt(path_to_file,delimiter="\t")
	return np.array(data[:,:-1],dtype=np.float), np.array(data[:,-1],dtype=np.int)


def conventional_writer(path_to_file,data,labels):
	data = np.concatenate((np.atleast_2d(data),np.reshape(labels,(-1,1))),axis=1)
	np.savetxt(path_to_file,data,delimiter="\t")

def _getModulePath():
	return os.path.dirname(__file__)


def load_aggregation():
	return (*conventional_reader(_getModulePath()+"\\Aggregation.txt"), "Aggregation dataset")

def load_compound():
	return (*conventional_reader(_getModulePath()+"\\Compound.txt"), "Compound")

def load_pathbased():
	return (*conventional_reader(_getModulePath()+"\\pathbased.txt"), "Pathbased")

def load_spiral():
	return (*conventional_reader(_getModulePath()+"\\spiral.txt"), "Spirals")

def load_D31():
	return (*conventional_reader(_getModulePath()+"\\D31.txt"), "example D31")

def load_R15():
	return (*conventional_reader(_getModulePath()+"\\R15.txt"), "example R15")

def load_Jain_toy_dataset():
	return (*conventional_reader(_getModulePath()+"\\jain.txt"), "Jain's toy datatset")

def load_flame():
	return (*conventional_reader(_getModulePath()+"\\flame.txt"), "Flame")

def load_birch1():
	return (*conventional_reader(_getModulePath()+"\\birch1.txt"), "Birch_1")

def load_birch2():
	return (*conventional_reader(_getModulePath()+"\\birch2.txt"), "Birch_2")

def load_EngyTime():
	return (*conventional_reader(_getModulePath()+"\\EngyTime.txt"), "EngyTime")

def load_Target():
	return (*conventional_reader(_getModulePath()+"\\Target.txt"), "Target")

def load_TwoDiamonds():
	return (*conventional_reader(_getModulePath()+"\\TwoDiamonds.txt"), "TwoDiamonds")

def load_WingNut():
	return (*conventional_reader(_getModulePath()+"\\WingNut.txt"), "WingNut")
