"""
Script to perform the cluster comparision analysis on the BrainNet data.
"""
############
# Standard #
############
import logging
import argparse
from pathlib import Path
import random
from datetime import datetime
from multiprocessing import Pool
from functools import partial

###############
# Third Party #
###############
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn import (metrics, cluster)
from tqdm import tqdm

##########
# Module #
##########
from r_functions import principal
from utils import (get_data, get_logger, histogram_data, histogram_data_3d)

# Globals
K = 6                           # Always running with 6 clusters
N_PCA = 3                       # Always assume pca to three
SIGFIGS = 4                     # Number of sigfigs to use when rounding

def setup_and_parse_args(parser, logger):
    """
    Parses the arguments and runs all the setup commands. More specifically, it:
    	- Adds all the standard BrainNet options
    	- Cluster comparison analysis specific options
    	- Sets the log level
    	- Checks that the standard arguments passed are valid
    	- Returns the args object
    """
    # # Add all the options
    # Standard
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase verbosity.",)
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in debug mode.",)        
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save the analysis graphs (cannot be used with " \
                        "plotting argument)")        
    parser.add_argument("-i", "--ipython", action="store_true",
                        help="Start an ipython shell at the end of the script.")
    parser.add_argument("--plt", action="store_true",
                        help="Plot the output graphs (cannot be used with " \
                        "saving argument)")

    # Analysis Specific
    parser.add_argument("--cc", action="store_true",
                        help="Run the cluster comparison analysis.")
    parser.add_argument("-n", action="store", type=int, default=100,
                        help="Number of times to rerun clustering algorithm.")
    
    # # Get the arguments passed
    args = parser.parse_args()

    # # Logging
    # Set the amount of logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logger.setLevel(log_level)
    logger.debug("Logging level set to {0}.".format(log_level))

    # # Run summary
    # Standard
    logger.debug("{0}ing the graphs when generated.".format(
        args.save * "Sav" or "Plott"))

    # # Analysis Specific
    if args.cc:
        logger.debug("Running cluster comparison analysis.")

    return args

def get_labels(data_path, k, data=None, clusterer=None):
    """
    Returns labels of the data.
    """
    if k == 6:
        # Return saved data if using k = 6
        labels_raw = np.array(get_data(data_path, ['CLU6_11']).values)
        return labels_raw.reshape(labels_raw.shape[0]) - 1
    elif data is not None and clusterer is not None:
        # Create new labels from the data
        return clusterer.fit_predict(data)
    else:
        logger.error("No labels saved for k={0} and no data or cluster " \
                     "provided to generate new labels.".format(args.k))
        raise ValueError

def compare_clusters(data, labels_true, clusterer, cluster_args, cluster_kwargs,
                     *args, **kwargs):
    """
    Compute the cluster solution using the inputted cluster algorithm and
    compare the results.
    """
    # Declare a fresh clusterer instance
    cluster_alg = clusterer(*cluster_args, **cluster_kwargs)
    # Perform the prediction
    labels_pred = cluster_alg.fit_predict(data)
    # Adjusted Rand Score
    ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi_score = metrics.normalized_mutual_info_score(labels_true, labels_pred)

    return ari_score, nmi_score

def run_analysis(data, args, clusterer, cluster_args, cluster_kwargs, n,
                 labels=None, pool=None, 
                 base_clusterer=cluster.AgglomerativeClustering(n_clusters=K)):
    """
    Function with the main analysis code.
    """
    # Create a new random seed
    np.random.seed()

    # Check if the data has been pca'd
    data_pca = data
    if data.shape[1] > N_PCA:
        data_pca = principal(data, N_PCA)

    # Get labels if they arent defined already
    if labels is None:
        try:
            labels = base_clusterer(n_components=int(args.k)).fit_predict(
                data_pca)
        except TypeError:
            labels = base_clusterer(n_clusters=int(args.k)).fit_predict(
                data_pca)

    if args.cc:
        # Run cluster stability analysis
        ari_scores, nmi_scores = zip(*(pool.map(partial(
            compare_clusters, data_pca, labels, clusterer, cluster_args,
            cluster_kwargs), tqdm(range(n)))))

    return np.mean(ari_scores), np.mean(nmi_scores)

def summarize_data(scores, n, name=""):
    """
    Prints the results in nicer way than just the dict repr.
    """
    out_str = "\nSummary of {0}\n".format(name or "results")
    out_str += "-"*len(out_str) + "\n"
    out_str += "Clustering algorithms run {0} times\n".format(n)
    out_str += "Mean score(s):\n"
    for key, score in scores.items():
        out_str += "    {0} : {1}\n".format(key, np.round(score, SIGFIGS))
    return out_str
    
if __name__ == "__main__":
    # # Basic Setup
    # Initialize the logger
    logger = get_logger(__name__)
    # Declare the argument parser
    parser = argparse.ArgumentParser(
        description="Cluster comparison on the BrainNet data.")
    # Set up parser and parse arguments
    args = setup_and_parse_args(parser, logger)
        
    # # Iterables that are looped through
    # Data Sets to run on
    data_sets = [
        "BRAINnet_partial.csv",
        ]
    # Columns to PCA
    columns = {
        # "all_items" : ["dass21_{0}".format(i+1) for i in range(21)],
        "pca" : ["Anhedonia_Factor","AnxArousal_Factor","Tension_Factor"],
        }

    # # Implemented Algorithms
    # Clustering Algorithms that can be chosen from
    algs_clustering = {
        "KMeans++" : cluster.KMeans,
        }
    # Clustering algorithm arguments
    args_clustering = {
        "KMeans++" : [],
        }
    # Clustering algorithm key word arguments
    kwargs_clustering = {
        "KMeans++" : {"n_clusters" : K, "init" : "k-means++"},
        }
    
    # Create the pool
    pool = Pool()

    # # Begin the pipline
    for data_set in data_sets:
        logger.info("Beginning cluster comparison analysis on '{0}' dataset"
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
            graph_folder = Path("graphs/comparisons/{0}/{1}".format(
                data_name, col))
            if not graph_folder.exists():
                graph_folder.mkdir(parents=True)

            # Container dictionary to hold the adjusted rand scores and mutual
            # info scores
            ari_scores = {}
            nmi_scores = {}
            # Define the base clustering algorithm to test against
            base_clusterer=cluster.AgglomerativeClustering(n_clusters=K)

            # Go through each Algorithm
            for key, alg in tqdm(algs_clustering.items()):
                logger.info("Getting scores for '{0}'".format(key))
                # Grab the arguments, use empty list if nothing was provided
                try:
                    c_args = args_clustering[key]
                except KeyError:
                    c_args = []
                # Same thing with the key word arguments
                try:
                    kw_args = kwargs_clustering[key]
                except KeyError:
                    kw_args = {}

                # Get the saved labels
                labels = get_labels(data_path, K, data=data,
                                    clusterer=base_clusterer)

                # Run the analysis
                ari_scores[key], nmi_scores[key] = run_analysis(
                    data, args, alg, c_args, kw_args, args.n, labels=labels,
                    pool=pool, base_clusterer=base_clusterer)

            print(summarize_data(ari_scores, args.n,
                                 name="Adjusted Rand Scores"))
            print(summarize_data(nmi_scores, args.n,
                                 name="Normalized Mutual Information Scores"))            
            
    # End Main Loop                    
    logger.info("Completed main analysis.")
    pool.close()
    pool.join()
    # # Last thing we do
    # Run an IPython shell to do some more analysis
    if args.ipython:
        import IPython; IPython.embed()
            
