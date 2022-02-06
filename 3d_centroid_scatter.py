# Lets import everything

############
# Standard #
############
import time
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
from functools import partial
from collections import Counter
from multiprocessing import Pool

###############
# Third Party #
###############
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import (cluster, manifold)
from sklearn.metrics import calinski_harabaz_score
from pyutils.pdutils import (GenerateMasterList, WriteTo)
from mpl_toolkits.mplot3d import Axes3D

##########
# Module #
##########
from r_functions import (principal, clusGap, scale)
from utils import (get_data, get_centers, closest_value, get_logger,
                   histogram_labels, histogram_data_3d)

logger = logging.getLogger(__name__)


# Define the analysis

def run_analysis(data, clusterer, n_samples, p, n_pca, k, *args):
    """
    Function with the main analysis code.
    """
    # Generate a new seed every iteration
    #np.random.seed()
    # Subsample the data
    data_subsample = data.sample(frac=p)

    labels = clusterer(n_clusters=int(k)).fit_predict(data_subsample)

    centroids = get_centers(data_subsample, labels)

    return centroids

# # Globals
percent = [.5]
n_samples = [10000]
load = True
save = False

# # Define Iterables that are looped through
# Data Sets to run on
data_sets = [
    "BRAINnet_partial.csv",
    ]
# Columns to PCA
columns = {
    "all_items" : ["dass21_{0}".format(i+1) for i in range(21)],
    }
# Clustering Algorithms to run analysis with
clustering_algs = {
    "agglomerative" : cluster.AgglomerativeClustering,
    }

# Load the data
if not load:
    # # Begin Pipeline
    # Grab the data
    for data_set in data_sets:
        logger.info("Beginning random sampling analysis on '{0}' dataset" \
                    "".format(data_set))
        data_path = Path('data/{0}'.format(data_set))
        # A string name to use as a reference for this dataset        
        data_name = str(data_path).split(".")[0].split("\\")[-1]

        # Loop through the desired slices of data
        for col in sorted(columns.keys()):
            logger.info("Using columns with key: '{0}.'".format(col))
            # Actually grab the dataset
            data = get_data(data_path, columns[col])
            # Place to save any generated plots
            graph_folder = Path("graphs/subsampling/{0}/{1}".format(
                data_name, col))
            if not graph_folder.exists():
                graph_folder.mkdir(parents=True)
                
            data_pca = principal(data, 3)

            # Loop through clustering algorithms
            for alg in sorted(clustering_algs.keys()):
                logger.info("Running analysis using '{0}' clustering.".format(
                    alg))
                # Loop through the number of subsamples
                for n in n_samples:
                    logger.info("Running analysis using {0} subsamples."
                                 "".format(n))
                    # Loop through all percentages
                    for p in percent:
                        logger.info("Using %{0} of data for subsamples." \
                                     "".format(np.round(100.*p)))
                        # Array to hold cluster centroids 
                        k_centroids = np.zeros((n, 6, 3))

                        # Begin the for loop through the n subsamples
                        logger.info("Beginning loop though subsamples.")
                        # Run the analysis
                        for i in tqdm(range(n)):
                            k_centroids[i,:,:] = run_analysis(
                                data_pca, clustering_algs[alg], n, p, 3, 6)
else:
    data_path = Path('data/{0}'.format("BRAINnet_partial.csv"))
    # A string name to use as a reference for this dataset        
    data_name = str(data_path).split(".")[0].split("\\")[-1]


if load:
    k_centroids_reshaped = np.array(GenerateMasterList(
        "data/centroids/k_centroids_reshaped.csv"))
else:
    # Reshape the centroids
    k_cent_y, k_cent_x, k_cent_z = k_centroids.shape
    # Reshape the centroid data to be reduced to 2D
    k_centroids_reshaped = k_centroids.reshape((
        k_cent_y*k_cent_x, k_cent_z))

# And see the head
k_centroids_reshaped.shape

if load:
    data_concat = np.array(GenerateMasterList(
        "data/centroids/data_concat.csv"))
    data_full_centroids = np.array(GenerateMasterList(
        "data/centroids/data_full_centroids.csv"))   
else:
    data_full_pca = get_data(data_path)
    data_full_labels = np.array(get_data(
        data_path, ['CLU6_11']).values-1).reshape(
            (data_full_pca.shape[0]))
    # Find the centroids
    data_full_centroids = get_centers(data_full_pca, 
                                      data_full_labels)
    # Concat all the data together so they are transformed 
    # the same way
    data_concat = np.concatenate((k_centroids_reshaped, 
                                  data_full_centroids), 
                                 axis=0)
# Verify we got the right shape again
data_concat.shape

# let's save the centroids
if not load and save:
    centroids_path = Path("data/centroids")
    if not centroids_path.exists():
        centroids_path.mkdir(parents=True)

    WriteTo(pd.DataFrame(k_centroids_reshaped),
            str(centroids_path / "k_centroids_reshaped.csv"))

    WriteTo(pd.DataFrame(data_concat),
            str(centroids_path / "data_concat.csv"))
    
    WriteTo(pd.DataFrame(data_full_centroids),
            str(centroids_path / "data_full_centroids.csv"))

# Define the colors to be used
color_list = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00"]
markers = ["d"]*6

# Set up the plotter
fig = plt.figure(dpi=600)
# fig.set_size_inches(7, 7)    
ax = fig.add_subplot(111, projection='3d')

# Plot the main data
xs = k_centroids_reshaped[:,0]
ys = k_centroids_reshaped[:,1]
zs = k_centroids_reshaped[:,2]
ax.scatter(xs, ys, zs, c="gray", marker=".", alpha=1/256, s=10)

# Plot the full data centroids
for i, (c, m) in enumerate(zip(color_list, markers)):
    print(c, m)
    xs = data_full_centroids[i,0]
    ys = data_full_centroids[i,1]
    zs = data_full_centroids[i,2]
    ax.scatter(xs, ys, zs, c=c, marker=m)
    
ax.set_xlabel('Anhedonia')
ax.set_ylabel('Anxious Arousal')
ax.set_zlabel('Tension')

ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
ax.set_zlim(-2, 4)

plt.show()
