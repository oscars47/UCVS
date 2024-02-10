#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:16:54 2020

@author: pfm
"""
import numpy as np
import time


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    #return np.power(cost, 1 / p)
    return np.abs(A-B)


def twed(A, timeSA, B, timeSB, nu=0.5, _lambda=1):
    # # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # # Compute Time Warp Edit Distance (TWED) for given time series A and B
    # #
    # # A      := Time series A (e.g. [ 10 2 30 4])
    # # timeSA := Time stamp of time series A (e.g. 1:4)
    # # B      := Time series B
    # # timeSB := Time stamp of time series B
    # # lambda := Penalty for deletion operation
    # # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # # Reference :
    # #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    # #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    # #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # # Check if input arguments
    # if len(A) != len(timeSA):
    #     print("The length of A is not equal length of timeSA")
    #     return None, None

    # if len(B) != len(timeSB):
    #     print("The length of B is not equal length of timeSB")
    #     return None, None

    # if nu < 0:
    #     print("nu is negative")
    #     return None, None

    # # Add padding
    # A = np.array([0] + list(A))
    # timeSA = np.array([0] + list(timeSA))
    # B = np.array([0] + list(B))
    # timeSB = np.array([0] + list(timeSB))

    # n = len(A)
    # m = len(B)
    # # Dynamical programming
    # DP = np.zeros((n, m))

    # # Initialize DP Matrix and set first row and column to infinity
    # DP[0, :] = np.inf
    # DP[:, 0] = np.inf
    # DP[0, 0] = 0

    # # Compute minimal cost
    # for i in range(1, n):
    #     for j in range(1, m):
    #         # Calculate and save cost of various operations
    #         C = np.ones((3, 1)) * np.inf
    #         # Deletion in A
    #         C[0] = (
    #             DP[i - 1, j]
    #             + Dlp(A[i - 1], A[i])
    #             + nu * (timeSA[i] - timeSA[i - 1])
    #             + _lambda
    #         )
    #         # Deletion in B
    #         C[1] = (
    #             DP[i, j - 1]
    #             + Dlp(B[j - 1], B[j])
    #             + nu * (timeSB[j] - timeSB[j - 1])
    #             + _lambda
    #         )
    #         # Keep data points in both time series
    #         C[2] = (
    #             DP[i - 1, j - 1]
    #             + Dlp(A[i], B[j])
    #             + Dlp(A[i - 1], B[j - 1])
    #             + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
    #         )
    #         # Choose the operation with the minimal cost and update DP Matrix
    #         DP[i, j] = np.min(C)
    # distance = DP[n - 1, m - 1]
    # return distance

    # mags_1: magnitudes for lightcurve 1
    # times_1: timestamps for lightcurve 1
    # mags_2: magnitudes for lightcurve 2
    # times_2: timestamps for lightcurve 2
    # nu: elasticity parameter (>= 0 for distance measure)
    # _lambda: cost of deletion

    # make sure data and timestamps have same length
    if len(A) != len(timeSA) or len(B) != len(timeSB):
        return None

    if nu < 0:
        return None

    # check for empty lightcurves
    if len(A) == 0 or len(B) == 0:
        return None
    

    # reindex
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)

    DP = np.zeros((n, m))
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # filling in the matrix
    for i in range(1, n):
        for j in range(1, m):
            # cost of operations list
            C = np.ones((3, 1)) * np.inf

            # deletion in lightcurve 1
            C[0] = (
                DP[i-1, j]
                + Dlp(A[i], A[i-1])
                + nu*(timeSA[i] - timeSA[i-1])
                + _lambda
            )

            # deletion in lightcurve 2
            C[1] = (
                DP[i, j-1]
                + Dlp(B[j], B[j-1])
                + nu*(timeSB[j] - timeSB[j-1])
                + _lambda
            )

            # keep both datapoints
            C[2] = (
                DP[i-1, j-1]
                + Dlp(A[i], B[j])
                + Dlp(A[i-1], B[j-1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )

            DP[i, j] = np.min(C)
    
    twed = DP[n - 1, m - 1]
    
    return twed

# Simple test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    A=np.array([0,0,1,1,2,3,5,2,0,1,-0.1])
    tA=list(i for i in range(len(A)))
    B=np.array([0,1,2,2.5,3,3.5,4,4.5,5.5,2,0,0,.25,.05,0])
    tB=list(i for i in range(len(B)))
    C=np.array([4,4,3,3,3,3,2,5,2,.5,.5,.5])
    tC=list(i for i in range(len(C)))
    nu=.1
    _lambda=.2

    t0 = time.time()
    print("twed(A,B,nu,lambda)=", twed(A,tA,B,tB,nu,_lambda))
    t1 = time.time()
    print (t1-t0)
    t0 = time.time()
    print("twed(A,C,nu,lambda)=", twed(A,tA,C,tC,nu,_lambda))
    t1 = time.time()
    print (t1-t0)
    t0 = time.time()
    print("twed(B,C,nu,lambda)=", twed(B,tB,C,tC,nu,_lambda))
    t1 = time.time()
    print (t1-t0)
        
    plt.plot(A, label='A')
    plt.plot(B, label='B')
    plt.plot(C, label='C')
    plt.legend()
    plt.show()