# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:35:24 2021

@author: Joan
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

# ----------------------------------------------------------------------
#  First the noiseless case
#X_ = np.random.random(5)*10
X_ = [0.80060977, 4.81696283, 4.45672956, 6.72472651, 4.48737259]
X = np.atleast_2d(X_).T


# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(2, (1e-2, 1e2)) #squared-exponential kernel with sigma2 scale factor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

# ----------------------------------------------------------------------
#Acquisition function
#PI
from scipy.stats import norm
max_y = max(y)
#max_y = max(y_pred)
Z = (y_pred-max_y)/sigma
epsilon = 0*max(sigma)
Z_ = (y_pred-max_y-epsilon)/sigma
PI = norm.cdf(Z_)
EI = (y_pred-max_y-epsilon)*norm.cdf(Z_)+sigma*norm.pdf(Z_)

UCB = y_pred + 3*sigma

plt.figure(figsize=(10,3))
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
    
plt.figure(figsize=(10,1.5))
plt.plot(x,PI,color='green')
plt.fill_between(x[:,0],PI,alpha=0.5,fc='g')
plt.xlabel('$x$')
plt.ylabel('$PI(x)$')

plt.figure(figsize=(10,1.5))
plt.plot(x,EI,color='green')
plt.fill_between(x[:,0],EI,alpha=0.5,fc='g')
plt.xlabel('$x$')
plt.ylabel('$EI(x)$')

plt.figure(figsize=(10,1.5))
plt.plot(x,UCB,color='green')
plt.fill_between(x[:,0],UCB,alpha=0.5,fc='g')
plt.xlabel('$x$')
plt.ylabel('$UCB(x)$')


# ----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel,# alpha=dy ** 2,
                              alpha = 2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')

plt.show()
