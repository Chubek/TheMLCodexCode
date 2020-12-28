#Simplified SMO
#http://cs229.stanford.edu/materials/smo.pdf
#Implemented by Chubak Bidpaa - December 2020

#Inputs:
#C: regularization parameter
#tol: numerical tolerance
#max-passes: maximum number of iterations over Larange multipliers without changing
#X, y: input data

#Output:
#alpha: Larange multipliers
#b: threshold for solution

#kernel used: no

import numpy as np
from chapter_01 import read_datasets
import random
import time
import math

dataset_reader_inst = read_datasets.DatasetReader("datasets/boston_housing.csv", ",")
X, y = dataset_reader_inst.read_into_array(["hello", "indus"], ["dev"])
C = 1..15
tol = 0.00001
max_passes = 100..150

def linear_classifier(alpha, X, X_i, y_i, b):
    return alpha * y_i * np.inner(X_i, X).sum() + b

def compute_L(alpha_i, alpha_j, y_i, y_j, C):
    if y_i != y_j:
        return max(0, alpha_i - alpha_j)
    else:
        return max(0, alpha_i + alpha_j - C)


def compute_H(alpha_i, alpha_j, y_i, y_j, C):
    if y_i != y_j:
        return min(C, C + alpha_i + alpha_j)
    else:
        return min(C, alpha_i + alpha_j)


def calculate_eta(X_i, X_j):
    i_j_inner = np.inner(X_i, X_j)
    i_i_inner = np.inner(X_i, X_i)
    j_j_inner = np.inner(X_j, X_j)

    return 2 * (i_j_inner - i_i_inner - j_j_inner)


def calculate_error(X_i, y_i, alpha, X, b):
    f_x_i = linear_classifier(alpha, X, X_i, y_i, b)
    E_i = f_x_i - y_i
    
    return E_i


def clip_alpha_j(alpha_j, H, L):
    if alpha_j > H:
        return H
    elif alpha_j < L:
        return L
    else:
        return alpha_j


def calculate_alpha_j(alpha_j, E_i, E_j, eta):
    return alpha_j - ((y_j * (E_i - E_j)) / eta)


def calculate_alpha_i(alpha_i, y_i, y_j, alpha_j_old, alpha_j):
    return alpha_i + (y_i * y_j) * (alpha_j_old - alpha_j)



def calculate_bs(b, X_i, y_i, alpha_i, alpha_j, E_i, E_j, X_j, y_j, alpha_i_old, alpha_j_old):

    b_one = (b - E_i - y_i) * (alpha_i - alpha_i_old) * np.inner(X_i, X_i) - (y_j * (alpha_j - alpha_j_old)) * np.inner(X_i, X_j) 
    b_two = (b - E_j - y_i) * (alpha_i - alpha_i_old) * np.inner(X_i, X_i) - (y_j * (alpha_j - alpha_j_old)) * np.inner(X_i, X_j) 

    return b_one, b_two

def compute_b(b_one, b_two, alpha_i, alpha_j, C):
    if alpha_i > 0 and alpha_i < C:
        return b1
    elif alpha_j > 0 and alpha_j < C:
        return b2
    else:
        (b_one + b_two) / 2


def rand_j(m, i):
    random.seed(time.time())
    
    j = random.randint(0, m)

    while i == j:
        j = random.randint(0, m)

    return j



def Simplified_smo(C=C, tol=tol, max_passes=max_passes, X=X, y=y):
    m = X.shape[0]    
    b = 0
    alphas = np.zeros(m)
    alphas_old = np.zeros(m)
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(1, m):
            E_i = calculate_error(X[i], y[i], alphas[i], X, b)
            y_i_E_i = y[i] * E_i

            if (y_i_E_i < (-1 * tol) and alpha[i] < C) or (y_i_E_i > tol and alpha[i] > 0):
                j = rand_j(m, i)
                E_j = calculate_error(X[j], y[j], alphas[j], X, b)
                alphas_old[i], alphas_old[j] = alphas[i], alphas[j]

                L = compute_L(alphas[i], alphas[j], y[i], y[j], C)
                H = compute_H(alphas[i], alphas[j], y[i], y[j], C)                

                if L == H:
                    continue

                eta = calculate_eta(X[i], X[j])

                if eta >= 0:
                    continue

                alphas[j] = clip_alpha_j(calculate_alpha_j(alpha[j], E_i, E_j, eta), H, L)

                if math.abs(alphas[j] - alphas_old[j]) < 0.000005:
                    continue

                alphas[i] = calculate_alpha_i(alphas[i], y[i], y[j], alphas_old[j], alphas[j])

                b_one, b_two = calculate_bs(b, X[i], y[i], alphas[i], alphas[j], E_i, E_j, X[j], y[j], alphas_old[i], alphas_old[j])

                b = compute_b(b_one, b_two, alphas[i], alphas[j], C)

                num_changed_alphas += 1
        
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        return alphas, b

                



