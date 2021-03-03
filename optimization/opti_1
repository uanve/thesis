# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:34:26 2021

@author: Joan
"""
import os
#cwd = os.getcwd()
#print("Current working directory: {0}".format(cwd))
os.chdir('C:/Users/Joan/OneDrive - Danmarks Tekniske Universitet/04 Forth Semester/thesis/code')

Delta_max = 10**10
u = 10
w_0 = 0.1

mu_1 = 10**-3
gamma = 0.9
gamma_inc = 1.2
e_c = 10**-6
tau = 0.1
d_ = 10**-2

#initialization
n_max = 10  # points to simulate
r = 10      # simulation runs x point

k = 0       # iteration index
n_0 = 1     # nº simulation runs
u_0 = 0     # trial points rejected

d0 = XXX #initial OD flows
Delta_0 = 10**3
f_a = sum(y_i - v_i)**2 by solving the metamodel
f(x) sum(y_i-E(y))

### solve (3) (12)
import numpy as np
import pandas as pd
from scipy.optimize import minimize,least_squares

df_DR = pd.read_csv("fict_network/DR.csv")
df_link_info = pd.read_csv("fict_network/link_info.csv")
df_RL = pd.read_csv("fict_network/RL.csv")
df_demand = pd.read_csv("fict_network/demand.csv")

k_jam = df_link_info["k_jam"]
q_cap = df_link_info["q_cap"]
n = df_link_info["n"]
v_max = df_link_info["v_max"]
l = df_link_info["l"]
RL = df_RL.to_numpy()
DR = df_DR.to_numpy()
demand = df_demand.to_numpy().reshape(100,)
I = 100
lam_0 = 100*np.random.rand(1000) 
y = lam_0 + 5*np.random.randn(1000) 
  

#k = c*k_jam*lam/(q_cap*n)
#v = v_max*(1-k/k_jam)
#t = l/v
#tr = np.dot(RL,t)
#P = np.exp(1.2*tr)/np.dot(DR.T,np.dot(DR,np.exp(1.2*tr)))
#lam = np.dot(RL.T,np.dot(np.identity(40)*P,x1))

beta_0 = 1
beta_1 = 0
beta_k = np.zeros(100)

def lam_(lam1):
    tr = np.dot(RL,l/(v_max*(1-(c*k_jam*lam1/(q_cap*n))/k_jam)))
    P = np.exp(1.2*tr)/np.dot(DR.T,np.dot(DR,np.exp(1.2*tr)))
    return np.dot(RL.T,np.dot(np.identity(40)*P,x1))

def f_a(lam2):
    return 1/I*np.sum((y-lam_(lam2))**2)

def tetha(x):
    return beta_1 + np.dot(beta_k,x[1000:])

def dd(d):
    return np.sum((d-demand)**2)


def obj_funct(x):
    return beta_0*f_a(x[:1000]) + tetha(x) + dd(x[1000:])


x = np.concatenate((lam_0,demand))

b = (0.0,100.0)
bnds = tuple([b for i in range(1100)])

solution = minimize(obj_funct,x,method="SLSQP",bounds=bnds,tol=1.0E-2)

x = solution.x
# show final objective
print('Final SSE Objective: ' + str(obj_funct(x)))



############### FIT METAMODEL #################
E_F = 100*np.random.rand(1000)  #simulation expected flow
fa = f_a(x[:1000])
tetha_ = tetha(x)
demand_ = x[1000:]
beta = np.concatenate(((beta_0,beta_1),beta_k))

def fun(beta):
    return (E_F-beta[0]*fa+tetha_)**2+0.001*((beta[0]-1)**2+np.dot(beta[1:],beta[1:]))


lsql = least_squares(fun, beta, bounds=([-5, 5]))
x = lsql.x

lsql.cost

lsql.optimality

fa = f_a(x[:1000])
