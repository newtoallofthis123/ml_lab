import numpy as np

# Declare the numpy array
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)

# Calculate the mean
m = np.mean(a.T, axis=1)
print(m)

# Center it
c = a - m
print(c)

# Calculate the covariance
v = np.cov(c.T)
print(v)

# eigen decompose
vals, vecs = np.linalg.eig(v)
print(vecs)
print(vals)

# project data
p = vecs.T.dot(c.T)
print(p.T)
