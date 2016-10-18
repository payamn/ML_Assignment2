#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001



# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
np.random.shuffle( data)
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.

t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]




DATA_FIG = 1

# Set up the slope-intercept figure

# Step size for gradient descent.
for eta in [0.5,0.3,0.1,0.05,0.01]:
  # Initialize w.
  w = np.array([0.1, 0, 0])

  # Error values over all iterations.
  e_all = []

  for iter in range (0,max_iter):
    y = sps.expit(np.dot(X[0, :], w))
    #e = -np.mean(np.multiply(t[0], np.log(y)) + np.multiply((1 - t[0]), np.log(1 - y)))
    e = 0
    for i in range (0,200):
      # Compute output using current w on all data X.


      y = sps.expit(np.dot(X[i,:],w))
      # e is the error, negative log-likelihood (Eqn 4.90)
      #if y==0:
      #  y+=0.00000001
      #if y==1:
      #  y-=0.00000001
      #y = np.clip(y, 1e-16, 1 - 1e-16)
      #e += (-np.mean(np.multiply(t[i],np.log(y)) + np.multiply((1-t[i]),np.log(1-y))))
      #print e
      # Add this error to the end of error vector.


      # Gradient of the error, using Eqn 4.91
      grad_e = np.multiply((y - t[i]), X[i,:])

      # Update w, *subtracting* a step in the error derivative since we're minimizing
      w_old = w
      w = w - eta*grad_e
    #e = e /X.shape[0]
    y = sps.expit(np.dot(X, w))
    y = np.clip(y, 1e-16, 1 - 1e-16)
    e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
      # Plot current separator and data.  Useful for interactive mode / debugging.
      # plt.figure(DATA_FIG)
      # plt.clf()
      # plt.plot(X1[:,0],X1[:,1],'b.')
      # plt.plot(X2[:,0],X2[:,1],'g.')
      # a2.draw_sep(w)
      # plt.axis([-5, 15, -10, 10])


      # Add next step of separator in m-b space.


      # Print some information.
    print 'epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T)

    e_all.append(e)
      # Stop iterating if error doesn't change more than tol.
    if iter>0:
      if np.absolute(e-e_all[iter-1]) < tol:
        break


  # Plot error over iterations
  plt.figure("error")
  plt.plot(e_all)
  plt.ylabel('Negative log likelihood')
  plt.title('Training logistic regression ')
  plt.xlabel('Epoch')
plt.legend([0.5,0.3,0.1,0.05,0.01])
plt.show()
