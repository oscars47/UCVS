import numpy as np
from matplotlib import pyplot as plt
from line_profiler_pycharm import profile
rng = np.random.default_rng()
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

def generate_data(num_clusters):
    X = make_blobs(n_samples=4000, centers=num_clusters)[0]
    return np.array([v[0] for v in X]), np.array([v[1] for v in X]), X


# https://www.researchgate.net/publication/272351873_NumPy_SciPy_Recipes_for_Data_Science_k-Medoids_Clustering
# noinspection PyUnboundLocalVariable

@profile
def k_medoids(D, k, data, x, y, max_steps=100,plot=False):
    """
    k-medoids clustering, given a matrix of pairwise distances and a number of desired clusters
    @param D: pairwise distance matrix
    @param k: number of desired clusters
    @param max_steps: optional maximum on number of iterations performed
    @return: matrix of medoid indices and a dictionary of {cluster id: data index}
    """
    m, n = D.shape
    if m != n:
        raise AttributeError("Distance matrix must be square")
    if k > m:
        raise AssertionError("Cannot have more medoids than data points")
    medoids = np.sort(rng.choice(m, k))  # choose k starting medoids
    next_medoids = np.copy(medoids)
    clusterDict = {}
    for i in range(max_steps):
        # find current clusters
        # (cluster for medoid M is all of the points whose closest medoid is M)
        medoidDistanceMatrix = D[:,medoids]  # a matrix with one column for each lightcurve and one row for each medoid,
        # where each entry is the distance between a lightcurve (column) and medoid (row)
        closestMedoidMatrix = np.argmin(medoidDistanceMatrix, axis=1)  #
        for j in range(k):
            clusterDict[j] = np.where(closestMedoidMatrix == j)[0]  # the jth cluster is the collection of points whose closest medoid has index j
        for j in range(k):
            # SUPER EXPENSIVE:
            submatrix = D[np.ix_(clusterDict[j], clusterDict[j])]  # submatrix of just the elements in cluster j

            meanDistances = np.mean(submatrix, axis=1)  # associate each lightcurve with its mean distance to the other lightcurves in the cluster
            newIdx = np.argmin(meanDistances)  # the new medoid is the one that is has the least mean distance to the other curves in the group
            newMedoid = clusterDict[j][newIdx]
            next_medoids[j] = newMedoid
        # if we choose the same medoids that we chose last time, we're done
        if np.array_equal(next_medoids, medoids):
            print(i)
            break
        if plot:
            for idxs, medoid, color in zip(clusterDict.values(), data[next_medoids], ["purple", "blue", "green"]):
                xData, yData = x[idxs], y[idxs]
                plt.scatter(xData, yData, s=1,c=color)
                plt.scatter(medoid[0], medoid[1], marker="X", s=20,c="red")
            plt.show()
            plt.clf()
        medoids = np.copy(next_medoids)
    else:
        print("No k-medoids convergence!")
        # we didn't break out of the for loop - i.e. we did not converge in time
        closestMedoidMatrix = np.argmin(medoidDistanceMatrix, axis=1)  #
        for j in range(k):
            clusterDict[j] = np.where(closestMedoidMatrix == j)[0]
            # the jth cluster is the collection of points whose closest medoid has index j
    return medoids, clusterDict


if __name__ == "__main__":
    num_clusters = 20
    for i in range(1):
        x, y, data = generate_data(num_clusters=num_clusters)
        distanceMatrix = pairwise_distances(data)
        medoids, clusterDict = k_medoids(distanceMatrix, num_clusters,data,x,y)
        plt.scatter(x, y)
        plt.show()
        plt.clf()
        cluster1 = clusterDict[0]
        dist2 = distanceMatrix[np.ix_(clusterDict[0], clusterDict[0])]
        print(dist2)
        medoids2, clusters2 = k_medoids(dist2, 3,data,x,y,plot=True)
        for idxs, medoid in zip(clusterDict.values(), data[medoids]):
            xData, yData = x[idxs], y[idxs]
            plt.scatter(xData, yData, s=1)
            plt.scatter(medoid[0], medoid[1], marker="X", s=20,c="red")
        for idxs, medoid in zip(clusters2.values(), data[medoids2]):
            xData, yData = x[idxs], y[idxs]
            plt.scatter(xData, yData, s=3)
            plt.scatter(medoid[0], medoid[1], marker="X", s=20, c="red")
        plt.show()
        plt.clf()
        for idxs, medoid,color in zip(clusters2.values(), data[medoids2],["purple","blue","green"]):
            xData, yData = x[idxs], y[idxs]
            plt.scatter(xData, yData, s=3,c=color)
            plt.scatter(medoid[0], medoid[1], marker="X", s=20, c="red")
        plt.show()