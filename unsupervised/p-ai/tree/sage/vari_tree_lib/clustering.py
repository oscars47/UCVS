import numpy as np
from matplotlib import pyplot as plt
from line_profiler_pycharm import profile
import networkx as nx
rng = np.random.default_rng()
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics.pairwise import pairwise_distances


def generate_blob_data(num_clusters,samples=5000):
    X = make_blobs(n_samples=samples, centers=num_clusters)[0]
    return np.array([v[0] for v in X]), np.array([v[1] for v in X]), X


def pairwise_distance_matrix(data):
    return np.linalg.norm(data[:, None, :] - data[None, :, :], axis=-1)


def generate_scattered_data(num_classes):
    X = make_classification(n_samples=4000, n_classes=num_classes)[0]
    return np.array([v[0] for v in X]), np.array([v[1] for v in X]), X


# https://www.researchgate.net/publication/272351873_NumPy_SciPy_Recipes_for_Data_Science_k-Medoids_Clustering
# noinspection PyUnboundLocalVariable
def kMedoids(D, k, data=None, x=None, y=None, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            print(f"Valenzuela et al. finished in {t} iterations")
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C


@profile
def k_medoids(D, k, max_steps=100, plot=False, data=None, x=None, y=None):
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
        medoidDistanceMatrix = D[:,
                               medoids]  # a matrix with one column for each lightcurve and one row for each medoid,
        # where each entry is the distance between a lightcurve (column) and medoid (row)
        closestMedoidMatrix = np.argmin(medoidDistanceMatrix, axis=1)  #
        for j in range(k):
            clusterDict[j] = np.where(closestMedoidMatrix == j)[
                0]  # the jth cluster is the collection of points whose closest medoid has index j
        for j in range(k):
            if not len(clusterDict[j]):
                # this is a pretty bad sign.
                # if we get an empty cluster, we change our medoid to the point farthest from our current medoid
                print("empty cluster!")
                farthestMedoid = np.argmax(D[j, :])
                next_medoids[j] = farthestMedoid
                continue
            # SUPER EXPENSIVE:
            submatrix = D[np.ix_(clusterDict[j], clusterDict[j])]  # submatrix of just the elements in cluster j

            meanDistances = np.mean(submatrix,
                                    axis=1)  # associate each lightcurve with its mean distance to the other lightcurves in the cluster
            newIdx = np.argmin(
                meanDistances)  # the new medoid is the one that is has the least mean distance to the other curves in the group
            newMedoid = clusterDict[j][newIdx]
            # move the medoid to the front of the array
            clusterDict[j][0], clusterDict[j][newIdx] = clusterDict[j][newIdx],clusterDict[j][0]
            next_medoids[j] = newMedoid
        # if we choose the same medoids that we chose last time, we're done
        if np.array_equal(next_medoids, medoids):
            print(f"Lina and Sage finished in {i} iterations")
            break
        if plot:
            for idxs, medoid, color in zip(clusterDict.values(), data[next_medoids], ["purple", "blue", "green"]):
                xData, yData = x[idxs], y[idxs]
                plt.scatter(xData, yData, s=1, c=color)
                plt.scatter(medoid[0], medoid[1], marker="X", s=20, c="red")
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


def find_distance(curves):
    return pairwise_distance_matrix(curves)


def make_tree(distance_matrix, cluster_indices, k, d,kmed=k_medoids):
    medoids_tree = {}
    clusters_tree = {}
    m, _ = distance_matrix.shape
    if m > k and d:
        medoids, clusters = kmed(distance_matrix, k)
        for cluster, medoid in zip(list(clusters.values()), medoids):
            # remove the medoid before continuing
            # cluster = np.delete(cluster, 0)
            dist_matrix = distance_matrix[np.ix_(cluster, cluster)]
            medoids_tree[medoid], temp_clusters = make_tree(dist_matrix, cluster, k, d - 1,kmed=kmed)
            clusters_tree[medoid] = [cluster, temp_clusters]
    else:
        clusters_tree = cluster_indices
    return medoids_tree, clusters_tree


# D: broken
def tree_to_string(tree_dict,key):
    string = key
    if not tree_dict:
        return string
    else:
        for key, d in tree_dict.items():
            string = string + "\t\n\t" + tree_to_string(d, key)
    return "\t"+string


def visualize_graph(graph_dict,title):
    G = nx.DiGraph()

    def add_nodes_and_edges(node, edges):
        G.add_node(node)
        for edge, sub_edges in edges.items():
            G.add_edge(node, edge)
            if sub_edges:
                add_nodes_and_edges(edge, sub_edges)

    for node, edges in graph_dict.items():
        add_nodes_and_edges(node, edges)

    center_node = list(graph_dict.keys())[0]

    # Use a spring layout for visualization
    print(center_node)
    pos = nx.spring_layout(G)
    colors = ['#71B6F4']*len(pos)
    colors[0] = '#71F4B0'
    displacement = {node: center_node_position - pos[center_node] for node, center_node_position in pos.items()}
    for node, position in pos.items():
        pos[node] = position + displacement[node]
    # Draw nodes and edges
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=colors, font_size=10,ax=ax)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from treelib import Node, Tree
    # g = {"a": {
    #     "b": {"d": {}, "e": {}},
    #     "c": {"f": {}}
    # }}
    # print(tree_to_string(g,"root"))
    # visualize_graph(g)
    num_clusters = 3
    x, y, data = generate_blob_data(num_clusters=num_clusters,samples=20)
    distanceMatrix = pairwise_distance_matrix(data)
    for func, name in zip([kMedoids, k_medoids], ["Lina and Sage k_medoids", "Valenzuela et al. k_medoids"]):
        medoids_tree, clusters_tree = make_tree(distanceMatrix, np.arange(len(x)), k=3, d=3)
        visualize_graph(medoids_tree, name)

    # print(tree)
    # tree.show()

    # for i in range(1):
    #     # x, y, data = generate_scattered_data(num_classes=num_clusters)
    #     x, y, data = generate_blob_data(num_clusters=num_clusters)
    #     # distanceMatrix = pairwise_distances(data)
    #     distanceMatrix = pairwise_distance_matrix(data)
    #     plt.scatter(x, y)
    #     plt.title("Un-Clustered Data")
    #     plt.show()
    #     plt.clf()
    #     for func, name in {k_medoids: "Lina and Sage", kMedoids: "Valenzuela et al."}.items():
    #
    #         medoids, clusterDict = func(distanceMatrix, num_clusters,data,x,y)
    #         # medoids, clusterDict = kMedoids(distanceMatrix, num_clusters)
    #
    #
    #         cluster1 = clusterDict[0]
    #         dist2 = distanceMatrix[np.ix_(clusterDict[0], clusterDict[0])]
    #         medoids2, clusters2 = k_medoids(dist2, 3,data,x,y,plot=True)
    #         for idxs, medoid in zip(clusterDict.values(), data[medoids]):
    #             xData, yData = x[idxs], y[idxs]
    #             plt.scatter(xData, yData, s=5)
    #             plt.scatter(medoid[0], medoid[1], marker="X", s=20,c="red")
    #         plt.title(name)
    #         for idxs, medoid in zip(clusters2.values(), data[medoids2]):
    #             xData, yData = x[idxs], y[idxs]
    #             plt.scatter(xData, yData, s=5)
    #             plt.scatter(medoid[0], medoid[1], marker="X", s=20, c="red")
    #         plt.show()
    #         plt.clf()
    #         for idxs, medoid,color in zip(clusters2.values(), data[medoids2],["purple","blue","green"]):
    #             xData, yData = x[idxs], y[idxs]
    #             plt.scatter(xData, yData, s=5,c=color)
    #             plt.scatter(medoid[0], medoid[1], marker="X", s=20, c="red")
    #         # plt.show()
