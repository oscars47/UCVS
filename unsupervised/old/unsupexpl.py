# file to try unsupervised clustering, implement the OPTICS algorithm
# @oscars47

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from collections import Counter
import plotly.express as px
from tqdm import tqdm

# set directories to normalized data
DATA_DIR = '/home/oscar47/Desktop/astro101/data/g_band/var_output/v0.1.1'
OUTPUT_DIR = os.path.join(DATA_DIR, 'unsupervised')
CLUSTER_DIR = os.path.join(OUTPUT_DIR, 'cluster')

# read in data with actual targets for verification
mm_n = pd.read_csv(os.path.join(DATA_DIR, 'mm_2_n_targ_var.csv'))
targets = mm_n['target'].to_list()
# now renove the final column of targets as well as first 2
mm_d = mm_n.iloc[:, 2:-1]

# import OPTICS package
from sklearn.cluster import OPTICS
# choose minimum number of samples as the number of dimensions in data + 1

def run(min_samples, percent, filename):
    # reduced data set
    r_index = int(percent*len(mm_d))
    mm_r = mm_d.iloc[:r_index, :].copy()
    # get actual classes for validation
    targets_r = targets[:r_index]
    print(len(mm_r))
    optics_clustering = OPTICS(min_samples=min_samples).fit(mm_r)
    labels = optics_clustering.labels_
    print('unique labels', set(labels))

    mm_r['prediction']=labels
    mm_r['targets'] = targets_r
    mm_r.to_csv(os.path.join(OUTPUT_DIR, filename))

# run optics on tsne or pca data
def run_optics_tsne(df, min_samples, outname):
    # save targets
    targets = df['targets'].to_list()
    # now remove them from the df
    df = df.iloc[:, :-1]

    # now call OPTICS
    optics_clustering = OPTICS(min_samples=min_samples).fit(df)
    labels = optics_clustering.labels_
    print('unique labels', set(labels))

    # add back in targets with predictions
    df['targets']=targets
    df['predictions']=labels

    # now save as csv
    df.to_csv(os.path.join(CLUSTER_DIR, outname))

# run dbscan on tsne or pca data
def run_dbscan_tsne(df, min_pts, outname):
    # save targets
    targets = df['targets'].to_list()
    # now remove them from the df
    df = df.iloc[:, :-1]

    # now call DBSCAN-----------------------
    #n_neighbors in sklearn.neighbors.NearestNeighbors to be equal to 2xN - 1, 
    nbrs = NearestNeighbors(n_neighbors=min_pts-1).fit(df)

    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(df)

    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    
    #now plot the kNN
    #look for the bend in the curve--this will give optimal value of epsilon
    #the elbow is thepoint farthest from a straight line

    k_dist = sort_neigh_dist[:, min_pts-2]
    
    #use kneed package to find knee points-----
    #https://github.com/arvkevi/kneed
    kn = KneeLocator(
    range(0, len(df)),
    k_dist,
    curve='convex',
    direction='increasing',
    interp_method='interp1d',
    )
    knee_value = kn.knee
    print('the elbow of the kNN curve occurs at: ', knee_value, k_dist[knee_value])
    
    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations")
    plt.title('kNN plot, elbow:' + str(knee_value))
    plt.savefig(os.path.join(CLUSTER_DIR, 'kNN_kk.jpeg'))
    
    #set value of epsilon
    e = k_dist[knee_value]
    
    #cluster!!------
    clusters = DBSCAN(eps=e, min_samples = min_pts).fit(df)
    # get cluster labels
    clusters_list = clusters.labels_
    
    # check unique clusters
    # -1 value represents noisy points could not assigned to any cluster
    unique_clusters = list(set(clusters.labels_))
    print('clusters:', unique_clusters)
    
    #get each cluster size
    Counter(clusters_list)
    
    #assign clusters to scaled data
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters_list
    #print(df)
    
    # print(df_clustered)

    # add back in targets with predictions
    df['targets']=targets
    df['predictions']=clusters_list

    # now save as csv
    df.to_csv(os.path.join(OUTPUT_DIR, outname))


# function to do unsupervised clustering but using PCA data
def run_PCA(min_samples, percent, filename):
    name = filename.split('.') # remove csv so we can concatenate pc

    mm_r = pd.read_csv(os.path.join(OUTPUT_DIR, name[0]+'_pc.csv'))

    r_index = int(percent*len(mm_d))
    targets_r = targets[:r_index]

    optics_clustering = OPTICS(min_samples=min_samples).fit(mm_r)
    labels = optics_clustering.labels_
    print('unique labels', set(labels))

    mm_r['prediction']=labels
    mm_r['targets'] = targets_r
    mm_r.to_csv(os.path.join(OUTPUT_DIR, filename))

# function to generate PCA
def get_PCA(filename):
    mm_r = pd.read_csv(os.path.join(OUTPUT_DIR, filename))

    # remove last 2 cols and perform 3 component PCA
    mm_r = mm_r.iloc[:, :-2]

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(mm_r)
    var = np.sum(pca.explained_variance_ratio_)

    print('variance', var)

    # get new df
    pc_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2', 'pc3'])
    name = filename.split('.') # remove csv so we can concatenate pc
    pc_df.to_csv(os.path.join(OUTPUT_DIR, name[0]+'_pc.csv'))

    return var # return the variance

# takes in df
def get_TSNE(mm_r, percent, perplexity, n_iterations, outname):
    # get subset of mm_r
    r_index = int(percent*len(mm_d))
    mm_r = mm_d.iloc[:r_index, :].copy()

    # get targets for validation
    targets_r = targets[:r_index]

    # perplexity parameter can be changed based on the input datatset
    # dataset with larger number of variables requires larger perplexity
    # set this value between 5 and 50 (sklearn documentation)
    # verbose=1 displays run time messages
    # set n_iter sufficiently high to resolve the well stabilized cluster
    # get embeddings
    tsne_arr = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iterations, verbose=1).fit_transform(mm_r)
    tsne_df = pd.DataFrame({'tsne1': [], 'tsne2': [], 'tsne3': []})

    for ls in tsne_arr:
        tsne_df = tsne_df.append({'tsne1': ls[0], 'tsne2':ls[1], 'tsne3':ls[2]}, ignore_index=True)
    
    tsne_df['targets'] = targets_r
    
    # print to verify
    print(tsne_df.head(10))

    # save
    print('saving!')
    tsne_df.to_csv(os.path.join(OUTPUT_DIR, outname))

# takes in df; removes indices that are SR or L
def get_TSNE_clipped(mm_r, percent, perplexity, n_iterations, outname):
    # get subset of mm_r
    r_index = int(percent*len(mm_d))

    # get targets for validation
    targets_r = targets[:r_index]
    mm_r = mm_d.iloc[:r_index, :].copy()

    mm_r['targets'] = targets_r
    mm_r = mm_r.loc[(mm_r['targets']!='SR') & (mm_r['targets']!='L')]
    # get new list of targets
    targets_new = mm_r['targets'].to_list()

    # now remove target column again
    mm_r = mm_r.iloc[:, :-1]
    print(len(mm_r))


    # perplexity parameter can be changed based on the input datatset
    # dataset with larger number of variables requires larger perplexity
    # set this value between 5 and 50 (sklearn documentation)
    # verbose=1 displays run time messages
    # set n_iter sufficiently high to resolve the well stabilized cluster
    # get embeddings

    tsne_arr = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iterations, verbose=1).fit_transform(mm_r)
    tsne_df = pd.DataFrame({'tsne1': [], 'tsne2': [], 'tsne3': []})

    for ls in tsne_arr:
        tsne_df = tsne_df.append({'tsne1': ls[0], 'tsne2':ls[1], 'tsne3':ls[2]}, ignore_index=True)
    
    tsne_df['targets'] = targets_new
    
    # print to verify
    print(tsne_df.head(10))

    # save
    print('saving!')
    tsne_df.to_csv(os.path.join(OUTPUT_DIR, outname))

# function to plot dimensionally-reduced data with complete filename----------
# name is whether we're running clipped or no clip
def plot_tsne_test(c_filename, name, percent, perplexity, n_iterations):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, c_filename))
    # get list of columns
    col_list = df.columns

    fig = px.scatter_3d(df, x='tsne1', y='tsne2', z='tsne3', color='targets')
    fig.update_layout(title="TSNE on " + str(np.round(percent*100, 2)) + " of " + name + " Data for Perplexity=" + str(np.round(perplexity, 2)) +  ", N_iterations=" + str(np.round(n_iterations, 2)), autosize=False, width=1000, height=1000)
    fig.show()

# function for visualizing results---------------------------
#c_alg: cluster algorithm
def plot_preds(c_filename, c_alg, percent, perplexity, n_iterations):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, c_filename))

    fig = px.scatter_3d(df,x='tsne1', y='tsne2', z='tsne3',
              color='targets')
    fig.update_layout(title=c_alg+"TARGET TSNE on " + str(np.round(percent*100, 2)) + " of " + name + " Data for Perplexity=" + str(np.round(perplexity, 2)) +  ", N_iterations=" + str(np.round(n_iterations, 2)), autosize=False,
                    width=1000, height=1000)
    fig.show()

    fig = px.scatter_3d(df, x='tsne1', y='tsne2', z='tsne3',
              color='predictions')
    fig.update_layout(title=c_alg+"PREDICTIONS TSNE on " + str(np.round(percent*100, 2)) + " of " + name + " Data for Perplexity=" + str(np.round(perplexity, 2)) +  ", N_iterations=" + str(np.round(n_iterations, 2)), autosize=False,
                    width=1000, height=1000)
    fig.show()

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

# function to visualize distribution of objects
def visualize_objs(percent):
    # get subset of mm_r
    r_index = int(percent*len(mm_d))
    # get targets for validation
    targets_r = targets[:r_index]
    # find unique objects, then get their respective counts
    targets_r_unique = list(set(targets_r))
    targets_r_unique_counts = []
    for targ in targets_r_unique:
        targets_r_unique_counts.append(targets_r.count(targ)) # append to list
    # now we can plot
    plt.figure(figsize=(10, 7))
    plt.bar(x=targets_r_unique, height=targets_r_unique_counts, color='magenta')
    addlabels(targets_r_unique, targets_r_unique_counts)
    plt.title('Number of objects per class in first 25%', fontsize=18)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Number of objects', fontsize=16)
    plt.show()

# percent=0.25
# # visualize_objs(percent)

# filename = 'mm_r_optics_.25_10.csv'
# filename_pca = 'mm_r_optics_.1_10_pca.csv'
# percent=0.1
# min_samples=10
# run(min_samples, percent, filename)
# var = get_PCA(filename)
# run_PCA(min_samples, percent, filename_pca)
# plot_preds(filename, var, percent)

# percent=.25
# perplexities=[200]
# n_iterations=[1000, 5000]
# for n_iter in tqdm(n_iterations):# loop through all possibilities
#     for perplex in tqdm(perplexities):
#         # compute tsne for both full and clipped data
#         outname_full = 'tsne_'+str(percent)+'_'+str(perplex)+'_'+str(n_iter)+'_2.csv'
#         outname_clip='tsne_'+str(percent)+'_'+str(perplex)+'_'+str(n_iter)+'_2_clipped.csv'
#         name_full = "Full2"
#         name_clip = "Clipped2"
#         # get_TSNE(mm_d, percent, perplex, n_iter, outname_full)
#         # get_TSNE_clipped(mm_d, percent, perplex, n_iter, outname_clip)

#         plot_tsne_test(outname_clip, name_full, percent, perplex, n_iter)
#         plot_tsne_test(outname_full, name_clip, percent, perplex, n_iter)


# going through specific files
# percent=.25
# files = ['tsne_0.25_200.csv', 'tsne_0.25_200_1000_2.csv', 'tsne_0.25_200_1000_clipped.csv', 'tsne_0.25_200_1000_2_clipped.csv', 'tsne_0.25_200_5000.csv', 'tsne_0.25_200_5000_2.csv', 'tsne_0.25_200_5000_clipped.csv', 'tsne_0.25_200_5000_2_clipped.csv']
# perplexities = [200 for i in range(8)]
# n_iterations = [1000 for i in range(4)] + [5000 for i in range(4)]
# names = ['Full1', 'Full2', 'Clipped1', 'Clipped2', 'Full1', 'Full2', 'Clipped1', 'Clipped2']
# for i in range(len(files)):
#     plot_tsne_test(files[i], names[i], percent, perplexities[i], n_iterations[i])

# computing OPTICS and DBSCAN
files = ['tsne_0.25_200.csv', 'tsne_0.25_200_1000_clipped.csv' , 'tsne_0.25_200_5000.csv', 'tsne_0.25_200_5000_clipped.csv']
names = ['Full', 'Clipped', 'Full', 'Clipped']
perplexity = 200
percent=.25
n_iterations =[1000, 1000, 5000, 5000]
min_pts = 6 # 2 * num features
for i, file in enumerate(files):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, file))
    outname_optics = 'optics_'+str(percent)+'_'+names[i]+'_'+str(perplexity)+'_'+str(n_iterations[i])+'.csv'
    outname_dbscan = 'dbscan'+str(percent)+'_'+names[i]+'_'+str(perplexity)+'_'+str(n_iterations[i])+'.csv'
    
    run_optics_tsne(df, min_pts, outname_optics)
    run_dbscan_tsne(df, min_pts, outname_dbscan)

    plot_preds(outname_optics, 'OPTICS', percent, perplexity, n_iterations[i])
    plot_preds(outname_dbscan, 'DBSCAN', percent, perplexity, n_iterations[i])

# for trying a range of perplexities with fixed n_iterations
# perplexity_ls = [30, 50, 75, 200, 500, 1000, 2000]
# percent = .25
# n_iterations = 5000
# for perplexity in perplexity_ls:
#     outname='tsne_'+str(percent)+'_'+str(perplexity)+'_'+str(n_iterations)+'.csv'
#     get_TSNE(mm_d, percent, perplexity, n_iterations, outname)
#     plot_tsne_test(outname, percent, perplexity, n_iterations)