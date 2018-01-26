import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import log_loss
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.stats import skellam

def oddspredict(fixtures, att_params, def_params, hmean, amean):
	resultodds = []
	neutralscore = (hmean+amean)/2
	for j in range(len(fixtures)):
		lamda = neutralscore * att_params[fixtures[j,0]] * def_params[fixtures[j,1]]
		mu = neutralscore * att_params[fixtures[j,1]] * def_params[fixtures[j,0]]
		p_hw, p_drw, p_aw = 0, 0, 0
		# calculate probability matrix
		for x in range(-75, 1):
			px = skellam.pmf(x, lamda, mu)
			if(x<0):
				p_aw = p_aw + px
			else:
				p_aw = p_aw + (px*0.5)
		resultodds.append(1-p_aw)
	return resultodds

def oddspredict2(fixtures, att_params, def_params, hmean, amean):
	resultodds = []
	neutralscore = (hmean+amean)/2
	for j in range(len(fixtures)):
		lamda = neutralscore * att_params[fixtures[j,0]] * def_params[fixtures[j,1]]
		mu = neutralscore * att_params[fixtures[j,1]] * def_params[fixtures[j,0]]
		px = skellam.cdf(-1, lamda, mu)
		p0 = skellam.pmf(0, lamda, mu)
		resultodds.append(px+p0*0.5)
	return resultodds

f1=np.array([(0,1),(2,3)])
f2=np.array([(1,0),(3,2)])
att_params = np.array([1.1,1.2,0.9,0.8])
def_params = np.array([1.15,1.25,0.92,0.98])
hmean = 68
amean = 64

a = oddspredict(f1, att_params, def_params, hmean, amean)
print(f1)
print(a)
a = oddspredict(f2, att_params, def_params, hmean, amean)
print(f2)
print(a)
a = oddspredict2(f1, att_params, def_params, hmean, amean)
print(f1)
print(a)
a = oddspredict2(f2, att_params, def_params, hmean, amean)
print(f2)
print(a)