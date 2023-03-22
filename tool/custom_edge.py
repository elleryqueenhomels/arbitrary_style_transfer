import argparse
import numpy as np
# from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("-ip", "--imagepath", help="Input image path")

args = parser.parse_args()


# vectorize to speed up
# def convolve2d(X, W):
# 	n1, n2 = X.shape
# 	m1, m2 = W.shape
# 	Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
# 	for i in range(n1):
# 		for j in range(n2):
# 			Y[i: i+m1, j: j+m2] += X[i, j] * W
# 	return Y

# change the sequence, even much faster
def convolve2d(X, W):
	n1, n2 = X.shape
	m1, m2 = W.shape
	Y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))
	for i in range(m1):
		for j in range(m2):
			Y[i: i+n1, j: j+n2] += W[i, j] * X
	return Y


# load the input image
img = mpimg.imread(args.imagepath)

# make it B&W
bw = img.mean(axis=2)

# Sobel operator - approximate gradient in X dir
Hx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

# Sobel operator - approximate gradient in Y dir
Hy = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
], dtype=np.float32)

t0 = datetime.now()
Gx = convolve2d(bw, Hx)
print('\nElapsed time:', (datetime.now() - t0))
plt.imshow(Gx, cmap='gray')
plt.show()

t0 = datetime.now()
Gy = convolve2d(bw, Hy)
print('\nElapsed time:', (datetime.now() - t0))
plt.imshow(Gy, cmap='gray')
plt.show()

# Gradient magnitude
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
plt.show()

# The gradient's direction
theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')
plt.show()
