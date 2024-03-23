import math
import os
import random
import sys
import timeit
import configparser

import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from line_profiler_pycharm import profile
# import networkx as nx

rng = np.random.default_rng()
# from sklearn.datasets import make_blobs, make_classification
# from sklearn.metrics.pairwise import pairwise_distances
from clustering import k_medoids, generate_blob_data, pairwise_distance_matrix, visualize_graph
import matplotlib.colors as mcolors

grandparentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(grandparentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
twed_mat_dir = config["twed_mat_dir"]

node_id_counter = -1


def subseq_node_dist(subsequence_idx, node_idx, dist_matrix):
    return dist_matrix[subsequence_idx][node_idx]


def choose_next_node(node1_idx, node2_idx, subseq_idx, dist_matrix):
    return node1_idx if subseq_node_dist(subseq_idx, node1_idx) <= subseq_node_dist(subseq_idx,
                                                                                    node2_idx) else node2_idx


def sum_record(records):
    record1 = records[0]
    for record2 in records[1:]:
        keys = set(list(record1.keys()).extend(list(record2.keys())))
        for key in keys:
            try:
                record1[key] += record2[key]
            except:
                try:
                    record1[key] = record2[key]
                except:
                    pass
        return record1


class Node:
    def __init__(self, depth, k, max_depth, medoid, cluster, dist_matrix):
        global node_id_counter
        node_id_counter += 1
        self.id = node_id_counter
        self.depth = depth
        self.k = k
        self.max_depth = max_depth
        self.medoid_idx = medoid
        self.cluster_indices = cluster
        self.dist_matrix = dist_matrix
        self._subseq_record = {}
        self.children = []
        self.is_leaf = False
        self.split_self()

    @property
    def subseq_record(self):
        if self.is_leaf:
            return self._subseq_record
        else:
            return sum_record([c.subseq_record for c in self.children])

    def split_self(self):
        if self.depth == self.max_depth or len(self.cluster_indices) < self.k:
            self.is_leaf = True
            return
        medoids_idx_list, clusters_indices_list = k_medoids(self.dist_matrix, self.k)

        for cluster_indices, medoid_idx in zip(list(clusters_indices_list.values()), medoids_idx_list):
            medoid_idx = self.cluster_indices[medoid_idx]
            dist_matrix = self.dist_matrix[np.ix_(cluster_indices, cluster_indices)]
            cluster_indices = self.cluster_indices[cluster_indices]
            newNode = Node(self.depth + 1, self.k, self.max_depth, medoid_idx, cluster_indices, dist_matrix)
            self.children.append(newNode)

    def as_dict(self, max_depth=-1):
        if max_depth != -1 and max_depth == self.depth:
            return {}
        child_dict = {}
        for child in self.children:
            child_dict[child.id] = child.as_dict(max_depth=max_depth)
        return child_dict

    def get_sub_leaves(self):
        if self.is_leaf:
            return [self]
        sub_leaves = []
        for child in self.children:
            sub_leaves.extend(child.get_sub_leaves())
        return sub_leaves


class Tree:
    def __init__(self, data, k, max_depth):
        self.data = data
        self.k = k
        self.max_depth = max_depth
        self.root = None
        self.dist_matrix = pairwise_distance_matrix(self.data)
        self.populate_tree()
        print("Done!")

    @property
    def leaves(self):
        def find_leaves(node):
            if node.is_leaf:
                return [node]
            leaves_below = []
            for child in node.children:
                leaves_below.extend(find_leaves(child))
            return leaves_below
        return find_leaves(self.root)


    def populate_tree(self):
        # call k medoids to find the root
        medoid_idx_list, cluster_indices_list = k_medoids(self.dist_matrix, k=1)
        medoid_idx, cluster_indices = medoid_idx_list[0], cluster_indices_list[0]
        self.root = Node(0, self.k, self.max_depth, medoid_idx, cluster_indices, self.dist_matrix)

    def nodes_at_depth(self, depth):
        def _rec_nodes_at_depth(node):
            if node.depth == depth:
                return [node]
            nodes_below_at_depth = []
            for child in node.children:
                nodes_below_at_depth.extend(_rec_nodes_at_depth(child))
            return nodes_below_at_depth

        return _rec_nodes_at_depth(self.root)

    def compare_subsequence(self,current_node,subseq_idx,lc_idx):
        if current_node.is_leaf:
            try:
                current_node.subseq_record[lc_idx] += 1
            except:
                current_node.subseq_record[lc_idx] = 1
            return current_node.id, current_node.medoid_idx
        sub_dist_mtrx = self.dist_matrix[np.array([c.medoid_idx for c in current_node.children]),subseq_idx]
        closest_child = current_node.children[np.argmin(sub_dist_mtrx)]
        return self.compare_subsequence(closest_child, subseq_idx, lc_idx)

def test_add():
    # NOTE: this test erroniously assumes that the correct behavior of the algorithm is for the subsequence to arrive at the leaf that is most similar to it - this is usually, but not always, how the algorithm works so this test is not accurate.  
    k = 3
    max_depth = 10
    samples = 20
    # x, y, alldata = generate_blob_data(num_clusters=k, samples=samples)
    alldata = np.loadtxt(os.path.join(twed_mat_dir, "20240302-1515")+".csv", delimiter=",")
    subdata = alldata[:5, :5]

    our_tree = Tree(subdata, k, max_depth)

    alldist = pairwise_distance_matrix(alldata)
    our_tree.dist_matrix = alldist

    leaf_idxs = np.array([leaf.medoid_idx for leaf in our_tree.leaves])
    leaf_mtrx = alldist[np.ix_(leaf_idxs, np.arange(samples))]
    correct = samples
    for i in range(500, samples):
        closest_mtrx_idx = np.argmin(leaf_mtrx[:, i])
        correct_idx = leaf_idxs[closest_mtrx_idx]
        tree_idx = our_tree.compare_subsequence(our_tree.root, i,0)
        if correct_idx != tree_idx[1]:
            correct -= 1
    print(f"Correct: {correct} out of {samples} ({round(correct/samples,2)*100}%)")



if __name__ == "__main__":
    test_add()

    exit()
    import matplotlib.animation as animation

    k = 3
    max_depth = 4
    x, y, data = generate_blob_data(num_clusters=k, samples=2000)

    our_tree = Tree(data, k, max_depth)
    d = {our_tree.root.id: our_tree.root.as_dict()}

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax1, ax2 = axes[0], axes[1]


    # create an animation slicing through different layers of the tree
    def update(frame):
        nodes = our_tree.nodes_at_depth(frame)
        ax1.cla()
        ax2.cla()
        s = []
        for node in nodes:
            xData = [c[0] for c in data[node.cluster_indices]]
            yData = [c[1] for c in data[node.cluster_indices]]
            s.append(ax1.scatter(xData, yData))
        d = {our_tree.root.id: our_tree.root.as_dict(max_depth=frame)}
        f, a = visualize_graph(d, title="Tree", fig=fig, ax=ax2)
        ax1.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax1.set_title("Cluster")
        plt.suptitle(f"Depth {frame}")
        s.append(a)
        return s


    ani = matplotlib.animation.FuncAnimation(fig=fig, func=update, frames=max_depth + 1, interval=1000, repeat=True)
    html = ani.to_jshtml()
    ani.save(filename="tree_slice.gif", writer="pillow")
    with open("tree_slice_ani.html", "w+") as f:
        f.write(html)
    print("Animated!")

    print(d)
    visualize_graph(d, "graph")

    # goal: plot k groups of leaves' clusters, each in a different color
    #  - get leaves
    #     - call get_sub_leaves on each of the root's k children
    #     - collect the returned results in a list of k lists
    #  - get a list of k clusters (lists of indices) from the list of k leaves
    #  - get a list of k lists of points from the list of k clusters
    #  - plot each of the k lists of points in a different color

    leaf_list_list = []
    for child in our_tree.root.children:
        leaf_list_list.append(child.get_sub_leaves())

    indices_list_list = []
    for leaf_list in leaf_list_list:
        indices_list = []
        for leaf in leaf_list:
            indices_list.append(leaf.cluster_indices)
        indices_list_list.append(indices_list)

    clusters_list_list = []
    for indices_list in indices_list_list:
        clusters_list = []
        for indices in indices_list:
            clusters_list.append(data[indices])
        clusters_list_list.append(clusters_list)

    fig, ax = plt.subplots()
    colors = list(mcolors.XKCD_COLORS)
    for clusters_list in clusters_list_list:
        for i, cluster in enumerate(clusters_list):
            xData = [c[0] for c in cluster]
            yData = [c[1] for c in cluster]
            ax.scatter(xData, yData, c=colors[i])
    # for i, clusters_list in enumerate(clusters_list_list):
    #     for cluster in clusters_list:
    #         xData = [c[0] for c in cluster]
    #         yData = [c[1] for c in cluster]
    #         ax.scatter(xData, yData, c=colors[i])
    plt.show()
