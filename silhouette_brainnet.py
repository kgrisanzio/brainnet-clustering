# Silhouette Scoring of BrainNet Data

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pyutils.pyutils import isiterable
from pyutils.pdutils import (GenerateMasterListWithHeader, WriteToWithHeader)
from sklearn.mixture import GMM
from sklearn import cluster
from sklearn.preprocessing import Imputer
from sklearn.metrics import calinski_harabaz_score
from IPython.core.debugger import Tracer
from sklearn.metrics import (silhouette_samples, silhouette_score,
                             adjusted_mutual_info_score)
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import (fcluster, linkage, ward)
_k = 0
_centroids = {_k : None}

class Centroids(object):
    k = 0
    centroids = {k:None}
    sim_scores = {}
    labels = {}
    @property
    def get_centroid(self):
        return self.centroids[self.k]
    
def get_data(columns=None):
    # Read in the data
    data = GenerateMasterListWithHeader(data_path)

    # Grab just the PCA data
    if columns is None or columns == "rad":
        try:
            data_slice = data[['Factor3_Anhedonia_PD_PTSD_MD_Allpts', 
                               'Factor3_AnxArousal_PD_PTSD_MD_Allpts', 
                               'Factor3_Irritabliity_PD_PTSD_MD_Allpts']]
        except KeyError:
            data_slice = data[['Anhedonia', 'Tension', 'Anxious Arousal']]
    else:
        data_slice = data[columns]

    data_slice_dropped = data_slice.dropna(axis=0, how='any').reset_index(
        drop=True)

    print("Number of rows in full dataset: {0}".format(len(data_slice)))
    print("Number of rows after dropping NaNs: {0}".format(len(
        data_slice_dropped)))
    print("Number of dropped rows: {0}".format(len(data_slice) - len(
        data_slice_dropped)))
    
    return data_slice_dropped

#Sklearn plotting as a function
def skplot(clustering, data, show=True, save=False, name=None, data_title=None,
           sigfigs=4):
    # Number of clusters
    num_clusters = int(clustering.max()) + 1

    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1] 
    ax.set_xlim([-0.5, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(data) + (num_clusters + 1) * 10])
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, clustering)
    print("For n_clusters =", num_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, clustering)

    y_lower = 10
    for i in range(num_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[clustering == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color_list = ["#E69F00", "#56B4E9", "#009E73",
                      "#F0E442", "#0072B2", "#D55E00"]
        # color = cm.spectral(float(i) / num_clusters)
        color = color_list[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, 
                          ith_cluster_silhouette_values,
                          facecolor=color, 
                          edgecolor=color, 
                          alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        
    plt.text(0.97, 0.03, "Mean Silhouette Coeff: {0:0<{width}}".format(
        np.round(silhouette_avg, sigfigs), width=sigfigs+2), ha='right', 
             va='center', transform=ax.transAxes)

    ax.set_title(("{0} Silhouette Analysis With {1} Clusters".format(
        data_title, num_clusters)), fontsize=14, fontweight='bold')
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    
    if show:
        plt.show()
    if save and name:
        if not name.parent.exists():
            name.parent.mkdir(parents=True)
        plt.savefig(str(name))
    plt.close('all')
    return silhouette_avg

def _get_centroids():
    return _centroids[_k]

def get_summary(summary_dict, title, xlabel, ylabel, save_path, color=None,
                dotted_line=None):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)
    for i, s in enumerate(sorted(summary_dict.keys())):
        if color is None:
            color = cm.spectral(float(i) / len(summary_dict.keys()))
        ax.plot(range_n_clusters, summary_dict[s], color=color, label=s,
                linewidth=1.33, marker="o", markersize=3.67)
        
    num_clusters = max(range_n_clusters)
    if len(summary_dict.keys()) > 1:
        plt.legend(loc='upper right')
    if dotted_line:
        ax.axvline(x=dotted_line, color="gray", linestyle="--", linewidth=1,
                   dashes=(5,8))        
    ax.set_title((title), fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(2, num_clusters + 2, 2))
    plt.savefig(save_path)
    plt.close('all')

def get_silhouette_summary(silhouette_dict, save_path):
    get_summary(silhouette_dict, "Average Silhouette Score vs K",
                "Number of Clusters K", "Average Silhouette Value", save_path)   

def get_calinski_summary(calinski_dict, save_path):
    get_summary(calinski_dict, "Calinski-Harabaz Statistic vs K",
                "Number of Clusters K", "Average Calinski-Harabaz Statistic",
                save_path)
    
def get_intra_distance_summary(intra_distance_dict, save_path, color=None):
    get_summary(intra_distance_dict, "Sum of Squared Errors vs K",
                "Number of Clusters K", "Sum of Squared Errors", save_path,
                color=color, dotted_line=6)

def get_variance_explained_summary(variance_dict, save_path):
    get_summary(variance_dict, "Variance Explained vs K", "Number of Clusters"
                " K", "Variance Explained", save_path)
            
def get_centers(df, labels):
    k = max(labels) + 1
    centers = np.zeros((k, df.shape[1])) 
    for i in range(k):
        centers[i, :] = df.iloc[labels==i, :].mean()
    return centers

def sum_squared_intra_cluster_differences(df, labels):
    centers = get_centers(df, labels)
    k = max(labels) + 1
    ave_intra_dist = [cdist(df.iloc[labels==j, :].values,
                            centers[i,:].reshape(1, centers.shape[1]))**2 for
                      i, j in enumerate(range(k))]
    return np.sum([np.sum(d) for d in ave_intra_dist])

def variance_explained(df, labels):
    return sum_squared_intra_cluster_differences(df, labels) / (2*len(df))                

def run_clustering(df, K, clusterer, append=False, silhouette=True, *args,
                   calinski=True, intra_distance=True, variance=False,
                   cent=None, **kwargs):
    alg_name = kwargs.pop("alg_name", None)
    data_title = kwargs.pop("data_title", None)
    show = kwargs.pop("show", False)
    save = kwargs.pop("save", False)
    if not isiterable(K):
        K = [K]
    print("\nRunning {0} Clustering on {1}".format(alg_name.title(), data_title))
    silhouette_averages = []
    calinski_scores = []
    intra_distances = []
    variances = []
    for k in K:
        # Set the current global k for seeded kmeans
        if alg_name == "k-means seeded":
            cent.k = k        
        clustering_args = {"k-means++" : {"init":"k-means++"},
                           "k-means seeded" : {"init":cent.get_centroid}}
        cluster_args = clustering_args.get(alg_name, {})  
                   
        # Cluster the data
        try:
            if alg_name == "agglomerative_sc_ward":
                labels = clusterer(ward(pdist(df, metric="euclidean")), t=k,
                                   criterion="maxclust") - 1
            elif alg_name == "agglomerative_sc_lnk":
                labels = clusterer(linkage(pdist(df, metric="sqeuclidean"),
                                           method="ward"), t=k,
                                   criterion="maxclust") - 1
            else:
                labels = clusterer(n_components=int(k),
                                   **cluster_args).fit_predict(df)
        except TypeError:
            labels = clusterer(n_clusters=int(k), **cluster_args).fit_predict(
                df)
        # import ipdb; ipdb.set_trace()
        # quit()
        if similarity and k == 6:
            # import ipdb; ipdb.set_trace()
            cent.sim_scores[alg_name] = adjusted_mutual_info_score(
                cent.truth.values[:,0], labels)
            cent.labels[alg_name] = labels
        # Save the centroid of clusters if this is agglomerative
        if alg_name == "agglomerative":
            cent.centroids[k] = get_centers(df, labels)

        if silhouette:
            silhouette_averages.append(skplot(labels, df, name=Path(
                "{0}/{1}/cluster_{2}.png".format(graph_folder, alg_name, k)),
                                              data_title=data_title, show=show,
                                              save=save, *args, **kwargs))
        if calinski:
            calinski_scores.append(calinski_harabaz_score(df, labels))
        if append:
            data_gen["{0}_k_{1}".format(alg_name, k)] = pd.Series(
                labels, index=df.index)
        if intra_distance:
            intra_distances.append(sum_squared_intra_cluster_differences(
                df, labels))
        if variance:
            variances.append(variance_explained(df, labels))
    return silhouette_averages, calinski_scores, intra_distances, None

if __name__ == "__main__":
    columns = {#"all_items" : ["dass21_{0}".format(i+1) for i in range(21)],
               "pca" : ["Anhedonia_Factor",
                        "AnxArousal_Factor",
                        "Tension_Factor"]}
    data_sets = ["BRAINnet_partial.csv",
#                 "BRAINnet_whole.csv",
#                 "RAD_partial.csv",
#                 "RAD_whole.csv",
                 ]
    
    # Save the plots
    save_plots = True
    # Show the plots being generated
    show_plots = False
    # Append clusters to data
    append = False
    # Save average silhouette score
    silhouette_summary = False
    # Run Calinski Metric
    calinski_summary = False
    # Run Silhouette scoring
    silhouette = False
    # Run Intra_Distance metric
    intra_distance_summary = True
    # Run Variance explained summary
    variance_summary = False
    # Print the similarity to ground truth for 6 clusters
    similarity = False

    generated_data = []
    silhouette_dict = {}
    calinski_dict = {}
    intra_distances_dict = {}
    variance_dict = {}
    cent = Centroids()
    
    for data_set in data_sets:
        # Data path
        data_path = Path('data/{0}'.format(data_set))
        _data_name = str(data_path).split(".")[0].split("\\")[-1]
        # Data name for the title of the graphs
        data_gen_path = Path(str(data_path).replace(".", "agg_gen."))

        if similarity:
            # Get Truth values
            cent.truth = get_data(['CLU6_11']) - 1
            # import ipdb; ipdb.set_trace()

        for col in columns.keys():
            data_title = "{0} {1}".format(_data_name, col).replace(
                "_", " ").title()
            graph_folder = Path("graphs/{0}/{1}/".format(_data_name, col))
            if not graph_folder.exists():
                graph_folder.mkdir(parents=True)
            data = get_data(columns[col])
            data_gen = data.copy()
            
            clustering_algs = {
#                "k-means" : cluster.KMeans,
                "agglomerative" : cluster.AgglomerativeClustering,
#                "agglomerative_sc_ward" : fcluster,
#                "agglomerative_sc_lnk" : fcluster,
#                "k-means++" : cluster.KMeans,
#                "k-means seeded" : cluster.KMeans
            }

            range_n_clusters = np.arange(2, 21)
            for alg in sorted(clustering_algs.keys()):
                key = "{0}_{1}_{2}".format(_data_name, col, alg)
                silhouette_dict[key], calinski_dict[key], \
                    intra_distances_dict[key], \
                    variance_dict[key] = run_clustering(
                        data, range_n_clusters, clustering_algs[alg],
                        show=show_plots, save=save_plots, alg_name=alg,
                        data_title=data_title, append=append,
                        silhouette=silhouette,
                        cent=cent)

        if append:
            WriteToWithHeader(data_gen, str(data_gen_path))

    color_list = ["#009E73"]

    if silhouette_summary:
        get_silhouette_summary(silhouette_dict,
                           save_path="graphs/silhouette_agglomerative.png")

    if calinski_summary:
        get_calinski_summary(calinski_dict,
                             save_path="graphs/calinski_agglomerative.png")

    if intra_distance_summary:
        for color in color_list:
            get_intra_distance_summary(
                intra_distances_dict, color=None,
                save_path="graphs/intra_distance_agglomerative.png")

    if similarity:
        for alg in sorted(cent.sim_scores.keys()):
            print("Similarity for {0}: %{1}".format(alg, cent.sim_scores[alg]*100))
    # if variance_summary:
    #     get_variance_explained_summary(variance_dict,
    #                       save_path="graphs/variance_explained_summary.png")

    
    # import IPython; IPython.embed()
