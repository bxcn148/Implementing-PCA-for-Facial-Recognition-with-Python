Download link :https://programming.engineering/product/implementing-pca-for-facial-recognition-with-python/

# Implementing-PCA-for-Facial-Recognition-with-Python
Implementing PCA for Facial Recognition with Python
Assignment Goals

Explore Principal Component Analysis (PCA) and the related Python packages (numpy, scipy, and matplotlib)

Make pretty pictures :)

Summary

In this project, you’ll be implementing a facial analysis program using PCA, using the skills from the linear algebra + PCA lecture. You’ll also continue to build your Python skills. We’ll walk you through the process step-by-step (at a high level).

Packages Needed for this Project

You’ll use Python 3 with the libraries NumPy, SciPy, and matplotlib (installation instructions linked). You should use a SciPy version >= 1.5.0 and the following import commands:

from scipy.linalg import eigh

import numpy as np

import matplotlib.pyplot as plt

Dataset

You will be using part of the Yale face dataset (processed). The dataset is saved in the ’YaleB_32x32.npy’ file. The ’.npy’ file format is used to store numpy arrays. We will test your code only using this provided dataset.

The dataset contains 2414 sample images, each of size 32 × 32. We will use n to refer to the number of images (so n = 2414) and d to refer to the number of features for each sample image (so d = 1024 = 32 ×32). Note, we’ll use xi to refer to the ith sample image which is a d-dimensional feature vector.

Program Specification

Implement these six Python functions in hw3.py to perform PCA on the dataset:

load_and_center_dataset(filename): load the dataset from the provided .npy file, center it around the origin, and return it as a numpy array of floats.

get_covariance(dataset): calculate and return the covariance matrix of the dataset as a numpy matrix (d × d array).

get_eig(S, m): perform eigendecomposition on the covariance matrix S and return a diagonal matrix (numpy array) with the largest m eigenvalues on the diagonal in descending order, and a matrix (numpy array) with the corresponding eigenvectors as columns.


get_eig_prop(S, prop): similar to get_eig, but instead of returning the first m, return all eigen-values and corresponding eigenvectors in a similar format that explain more than a prop proportion of the variance (specifically, please make sure the eigenvalues are returned in descending order).

project_image(image, U): project the image into the m-dimensional subspace and then project back into d × 1 dimensions and return that.

display_image(orig, proj): use matplotlib to display a visual representation of the original image and the projected image side-by-side.

5.1 Load and Center the Dataset ([20] points)

You’ll want to use the the numpy function load() to load the YaleB_32x32.npy file into Python.

x = np.load(filename)

This should give you a n × d dataset (recall that n = 2414 is the number of images in the dataset and d = 1024 is the number of dimensions of each image). In other words, each row represents an image feature vector.

Your next step is to center this dataset around the origin. Recall the purpose of this step from lecture

— it is a technical condition that makes it easier to perform PCA, but it does not lose any important information.

To center the dataset is simply to subtract the mean µx from each data point xi (image, in our case), i.e., xcenti = xi − ux, where

1

n

µx =

xi.

n

i=1

You can take advantage of the fact that x (as defined above) is a numpy array and, as such, has this convenient behavior:

x = np.array([[1,2,5],[3,4,7]])

np.mean(x, axis=0)

array([2., 3., 6.])

x – np.mean(x, axis=0) array([[-1., -1., -1.],

[ 1., 1., 1.]])

After you’ve implemented this function, it should work like this:

x = load_and_center_dataset(‘YaleB_32x32.npy’)

len(x)

2414

len(x[0])

1024

np.average(x)

-8.315174931741023e-17

(Its center isn’t exactly zero, but taking into account precision errors over 2414 arrays of 1024 floats, it’s what we call “close enough.”)

From now on, we will use xi to refer to xcenti.

5.2 Find the Covariance Matrix ([15] points)

Recall, from lecture, that one of the interpretations of PCA is that it is the eigendecomposition of the sample covariance matrix. We will rely on this interpretation in this assignment, with all of the information you need below.

5.4 Get all Eigenvalues/Eigenvectors that Explain More than a Certain Pro-portion of the Variance ([8] points)

Instead of taking the top m eigenvalues/eigenvectors, we may want all the eigenvectors that explain more than a certain proportion of the variance. Let λi be an eigenvalue of the covariance matrix S. Then λi’s corresponding eigenvector’s proportion of variance is

λi

n .

λ

Return the eigenvalues as a diagonal matrix, in descending order, and the corresponding eigenvectors as columns in a matrix. Hint: subset_by_index was useful for the previous function, so perhaps something similar could come in handy here. What is the trace of a matrix?

Again, make sure to return the diagonal matrix of eigenvalues first, then the eigenvectors in corresponding columns. You may have to rearrange the output of eigh to get the eigenvalues in decreasing order and make sure to keep the eigenvectors in the corresponding columns after that rearrangement.

Lambda, U = get_eig_prop(S, 0.07)

print(Lambda)

[[1369142.41612494

0.

]

[

0.

1341168.50476773]]

print(U)

[[-0.01304065 -0.0432441 ]

[-0.01177219 -0.04342345]

[-0.00905278 -0.04095089]

…

0.00148631 0.03622013]

0.00205216 0.0348093 ]

0.00305951 0.03330786]]

5.5 Project the Images ([15] points)

Given an image in the dataset and the eigenvectors from get_eig (or get_eig_prop), compute the PCA representation of the image.

Let uj represent the jth column of U. Every uj is an eigenvector of S with size d × 1. If U has m eigenvectors, the image xi is projected into an m dimensional subspace. The PCA projection represents images as a weighted sum of the eigenvectors. This projection only needs to store the weight for each

eigenvector (

m dimensions) instead of the entire image (d-dimensions). The projection α

i ∈ R

m is computed

–

T

such that αij = uj xi.

m

The full-size representation of αi can be computed as xpca

=

αij uj . Notice that each eigenvector

i

j=1

uj is multiplied by its corresponding weight αij . The reconstructed image, xpcai, will not necessarily equal the original image because of the information lost projecting xi to a smaller subspace. This information loss will increase as less eigenvectors are used. Implement project_image to compute xpcai for an input image.

projection = project_image(x[0], U)

print(projection)

[6.84122225 4.83901287 1.41736694 … 8.75796534 7.45916035 5.4548656 ]

5.6 Visualize ([25] points)

We’ll be using matplotlib’s imshow.

Follow these steps to visualize your images:

Reshape the images to be 32 × 32 (before this, they were being thought of as 1 dimensional vectors in

R1024).


Create a figure with one row of two subplots.

Title the first subplot (the one on the left) as “Original” (without the quotes) and the second (the one on the right) as “Projection” (also without the quotes).

Use imshow with the optional argument aspect=’equal’

Use the return value of imshow to create a colorbar for each image.

Render your plots!

Below is a simple snippet of code for you to test your functions. Do not include it in your submission!

x = load_and_center_dataset(‘YaleB_32x32.npy’)

S = get_covariance(x)

Lambda, U = get_eig(S, 2)

projection = project_image(x[0], U)

display_image(x[0], projection)


Submission Notes

Please submit your files in a .zip archive named hw3_<netid>.zip, where you replace <netid> with your netID (i.e., your wisc.edu login). Inside your zip file, there should be only one file named hw3.py. Do not submit a Jupyter notebook .ipynb file.

Be sure to remove all debugging output before submission; failure to do so will be penalized ([10] points):

Your functions should run silently (except for the image rendering window in the last function).

No code should be put outside the function definitions (except for import statements; helper functions are allowed).

ALL THE BEST!

5

