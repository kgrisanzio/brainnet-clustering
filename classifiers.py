"""
Script to perform the classifier analysis on the BrainNet data.
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
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

##########
# Module #
##########
from r_functions import principal
from utils import (get_data, get_logger, histogram_data, histogram_data_3d)

def setup_and_parse_args(parser, logger):
    """
    Parses the arguments and runs all the setup commands. More specifically, it:
    	- Adds all the standard BrainNet options
    	- Classifier analysis specific options
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
    parser.add_argument("-n", "--n_subsamples", action="store", nargs="+",
                        type=int, default=(100,),
                        help="Number of times to rerun subsamples.")
    parser.add_argument("-p", "--percent", action="store", nargs="+",
                        type=float, default=(.67,),
                        help="Percent of data to use per subsample.")
    parser.add_argument("--n_pca", action="store", type=int,
                        default=3,
                        help="Number of components to keep with PCA.")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save the analysis graphs (cannot be used with " \
                        "plotting argument)")        
    parser.add_argument("-i", "--ipython", action="store_true",
                        help="Start an ipython shell at the end of the script.")
    parser.add_argument("--plt", action="store_true",
                        help="Plot the output graphs (cannot be used with " \
                        "saving argument)")

    # Analysis Specific
    parser.add_argument("-k", action="store", type=int,
                        default=6,
                        help="Cluster number to perform analysis on.")
    parser.add_argument("--cs", action="store_true",
                        help="Run the cluster stability analysis.")
    parser.add_argument("--cs_classifier", action="store", type=str,
                        default="LDA",
                        help="The classifier alg to use for the cluster " \
                        "stability analysis.")
    parser.add_argument("--test", action="store_true",
                        help="Run the classifier analysis on just the testing "
                        "data results.")
    
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
    logger.debug("Running analysis on {0} subsamples.".format(
        args.n_subsamples))
    args.percent = [p / 100 if 1 < p <= 100 else p for p in args.percent]
    for p in args.percent:
        if p <= 0 or p > 100:
            logger.warning("Invalid entry for percent to subsample, {0}" \
                            "".format(p))
    args.percent = [.5 if p <= 0 or p > 100 else p for p in args.percent]
    logger.debug("Running analysis with {0} subsample(s) size.".format(
        args.percent))
    logger.debug("Running PCA keeping top {0} components".format(
        args.n_pca))
    logger.debug("{0}ing the graphs when generated.".format(
        args.save * "Sav" or "Plott"))

    # Analysis Specific
    logger.debug("Running the analysis with k = {0}.".format(args.k))
    if args.cs:
        logger.debug("Running cluster stability analysis using the '{0}' " \
                     "classification algorithm.".format(args.cs_classifier))
    return args

def histogram_data_cs(ari_scores, nmi_scores, com_scores, hom_scores,
                      n_subsamples=None, percent=None, k=None, plot=True,
                      save=False):
    """
    Configures and plots the histogrms of the resultsd from the cluster
    stability analysis.
    """
    # Fields common to all graphs
    xlabel = "Scores"
    ylabel = "Number of Occurences"
    title = "Histogram of "
    text = ""
    can_save = False

    
    if save and percent and n_subsamples and k:
        save_dir = Path("graphs/histograms_cs/{0}_k_{1}_p_{2}_subsamples/" \
                        "".format(k, percent, n_subsamples))
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        can_save = True         # We have all the necessary components to save

    # Alert user if saving was requested but we cannot save
    if save and not can_save:
        logger.warn("Cannot save images. Graph saving was requested but not " \
                    "all arguments to properly save were provided.")

    # Declare the iterables
    scores = [ari_scores, nmi_scores, com_scores, hom_scores]
    score_titles = ["Adjusted Rand Scores",
                    "Normalized Mutual Information Scores",
                    "Completeness Scores",
                    "Homogeneity Scores"]

    # Loop through the scores
    for score, score_title in zip(scores, score_titles):
        text = "Std Score: {0}\n".format(np.round(score.std(), 4))
        text += "Range Score: {0}\n".format(np.round(score.max()-score.min(),4))
        
        # Plot the Graphs
        if plot:
            histogram_data(
                score, xlabel=xlabel, ylabel=ylabel,
                title=title+"{0}".format(score_title),
                mean_line=True,                
                text="Mean Score: {0}\n".format(
                    np.round(score.mean(), 4)) + text
            )
            plt.show()
            
        # And/Or Save them
        if can_save:
            histogram_data(
                score, xlabel=xlabel, ylabel=ylabel,
                title=title+"{0}".format(score_title),
                mean_line=True,
                text="Mean Score: {0}\n".format(
                    np.round(score.mean(), 4)) + text,
            )
            plt.savefig(
                str(save_dir) + "/" + score_title.replace(" ","_").lower() + \
                ".png")
            
    # Cleanup
    plt.close('all')

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
    
def cluster_similarity(X_train, X_test, y_train, y_test, classifier=None,
                       clusterer=None, test=False):
    """
    Run pipeline that computes the cluster similarity by returning the adjusted
    rand index, normalized mutual information score, completeness score and
    homogeneity score of the clustering solution obtained from training on
    the cluster solution of X_train, and combining with that solution with the
    classifer results on X_test.
    """
    # # Preliminary Checks
    if classifier is None:
        # sklearn Linear Disciminant Classifier
        classifier = LinearDiscriminantAnalysis(solver='svd')
    if clusterer is None:
        # Agglomerative clustering using Wards method
        clusterer = cluster.AgglomerativeClustering(
            n_clusters=max(labels_true)+1)

    # # Get the relevant data
    # Obtain a set of labels from the clusterer
    labels_X_train_cluster = clusterer.fit_predict(X_train)
    # Train the classifier on X_train and these labels
    classifier.fit(X_train, labels_X_train_cluster)
    
    # Obtain labels for the testing set
    if isinstance(classifier, LinearDiscriminantAnalysis):
        # Using LDA which has a slightly different API
        labels_X_test_classifier_raw = classifier.decision_function(X_test)
        # Get label idx with highest confidence
        labels_X_test_classifier = np.argmax(labels_X_test_classifier_raw,
                                              axis=1)
    else:
        labels_X_test_classifier = classifier.transform(X_test)

    if test:
        labels_pred = labels_X_test_classifier
        labels_true = y_test
    else:
        # Concat these labels to form the overall cluster solution
        labels_pred = np.concatenate(
            (labels_X_train_cluster, labels_X_test_classifier), axis=0)
        # Concat the testing labels into one larger array of labels
        labels_true = np.concatenate((y_train, y_test), axis=0)

    # # Compute the various metrics
    # Adjusted Rand Index
    ari_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    # Normalized Mutual Information
    nmi_score = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    # Completeness Score
    com_score = metrics.completeness_score(labels_true, labels_pred)
    # Homogeneity Score
    hom_score = metrics.homogeneity_score(labels_true, labels_pred)

    return ari_score, nmi_score, com_score, hom_score

def run_analysis(data, args, clusterer, labels=None, percent=.67, *pargs):
    """
    Function with the main analysis code.
    """
    # Create a new random seed
    np.random.seed()
    
    # Data containers
    ari_score = None            # Adjusted rand index
    nmi_score = None            # normalized mutual information score
    comp_score = None           # Completeness score
    hom_score = None            # Homogeneity score

    # Check if the data has been pca'd
    data_pca = data
    if data.shape[1] > args.n_pca:
        data_pca = principal(data, args.n_pca)

    # Get labels if they arent defined already
    if labels is None:
        try:
            labels = clusterer(n_components=int(args.k)).fit_predict(data_pca)
        except TypeError:
            labels = clusterer(n_clusters=int(args.k)).fit_predict(data_pca)
            
    # Shuffle the data and split into the training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        data_pca, labels, random_state=random.seed(datetime.now()),
        train_size=percent)

    if args.cs:
        # # Run cluster stability analysis
        # # Concat the testing labels into one larger array of labels
        # labels_true = np.concatenate((y_train, y_test), axis=0)
        # Run the analysis
        ari_score, nmi_score, comp_score, hom_score = cluster_similarity(
            X_train, X_test, y_train, y_test, classifier=args.cs_classifier,
            clusterer=clusterer, test=args.test)

    return ari_score, nmi_score, comp_score, hom_score

if __name__ == "__main__":
    # # Basic Setup
    # Initialize the logger
    logger = get_logger(__name__)
    # Declare the argument parser
    parser = argparse.ArgumentParser(
        description="Classifer analysis on subsets of the BrainNet data.")
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
        "agglomerative" : cluster.AgglomerativeClustering(n_clusters=args.k),
        }
    # Classifier Algorithms that can be chosen from
    algs_classifier = {
        "LDA" : LinearDiscriminantAnalysis(solver='svd'),
        }
    
    if args.cs:        
        # Set the cs_classifier to be the actual classifer object
        try:
            args.cs_classifier = algs_classifier[args.cs_classifier]
        except KeyError:
            # Handle invalid entries
            logger.warn("Invalid cluster stability classifer inputted '{0}'. " \
                        "Valid entries are: {1}. Using 'LDA.'".format(
                            args.cs_classifier, sorted(algs_classifier.keys())))
            args.cs_classifier = algs_classifier["LDA"]

    # Create the pool
    pool = Pool()
        
    # # Begin the pipline
    for data_set in data_sets:
        logger.info("Beginning classifier analysis on '{0}' dataset".format(
            data_set))
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
        
            # Loop through the number of subsamples
            for n in args.n_subsamples:
                logger.info("Running analysis using {0} subsamples.".format(n))
        
                # Loop through all percentages
                for p in args.percent:
                    logger.info("Using %{0} of data for subsamples.".format(
                        np.round(100.*p)))
                    # # Data containiners
                    ari_scores = np.zeros((n))
                    # Normalized Mutual Information
                    nmi_scores = np.zeros((n))
                    # Completeness Score
                    com_scores = np.zeros((n))
                    # Homogeneity Score
                    hom_scores = np.zeros((n))

                    # # Get labels for the data
                    labels = get_labels(data_path, args.k, data=data,
                                        clusterer=algs_clustering[
                                            "agglomerative"])

    			    # Begin the for loop through the n subsamples
                    logger.info("Beginning loop though subsamples.")
                    # for i in tqdm(range(n)):
                    # import ipdb; ipdb.set_trace()
                    ari_scores, nmi_scores, com_scores, hom_scores = zip(
                        *(pool.map(partial(
                            run_analysis, data, args,
                            algs_clustering["agglomerative"], labels, p),
                                   tqdm(range(n)))))

                    # Turn the results into arrays
                    ari_scores = np.array(ari_scores)
                    nmi_scores = np.array(nmi_scores)
                    com_scores = np.array(com_scores)
                    hom_scores = np.array(hom_scores)
                    
                    # # Post Processing and Graphing
                    # Plotting and Saving
                    if args.cs and (args.plt or args.save):
                        histogram_data_cs(
                            ari_scores, nmi_scores, com_scores, hom_scores,
                            n_subsamples=n, percent=p, plot=args.plt,
                            save=args.save, k=args.k)

    # End Main Loop                    
    logger.info("Completed main analysis.")
    pool.close()
    pool.join()

    # # Last thing we do
    # Run an IPython shell to do some more analysis
    if args.ipython:
        import IPython; IPython.embed()
