from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import torch
import dgl
from sklearn.decomposition import PCA


def create_knn_adj_mat(features, k, weighted=False, n_jobs=None, algorithm='auto', threshold=None):
    """
    Create a directed normalized adjacency matrix from input nodes based on k-nearest neighbours
    Parameters:
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        weighted (bool): set to True for weighted adjacency matrix (based on Euclidean distance)
        n_jobs (int): number of jobs to deploy
        algorithm (str): Choose between auto, ball_tree, kd_tree or brute
        threshold (float): Cutoff value for the Euclidean distance
    Returns:
        (coo matrix): adjacency matrix as a sparse coo matrix
    """

    # initialize and fit nearest neighbour algorithm
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, algorithm=algorithm)
    neigh.fit(features)

    if weighted:
        # Obtain matrix with distance of k-nearest points
        adj_mat_weighted = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='distance')).toarray())

        if threshold:
            indices_to_zero = adj_mat_weighted > threshold
            adj_mat_weighted[indices_to_zero] = 0

        # Take reciprocal of non-zero elements to associate lower weight to higher distances
        non_zero_indices = np.nonzero(adj_mat_weighted)
        adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

        # Normalize rows
        coo_matrix = sp.coo_matrix(adj_mat_weighted)
        normalized_coo_matrix = normalize(coo_matrix)

        return normalized_coo_matrix

    # Obtain the binary adjacency matrix
    adj_mat_connectivity = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='connectivity')).toarray())

    # Remove self-connections
    adj_mat_connectivity -= np.eye(features.shape[0])

    return sp.coo_matrix(adj_mat_connectivity)


def draw_graph(adj_mat, label, idx, weighted=False, directed=False):
    """
    Plots the graph for a weighted adjacency matrix
    Parameters:
        adj_mat (sparse coo matrix): Weighted adjacency matrix
        label (bool): Indicates the label for the graph to be drawn (used in saved file name)
        idx (int): Indicates the index for the graph to be drawn (used in saved file name)
        weighted (bool): Indicates whether the graph edges are weighted or not
        directed (bool): Indicates whether the graph is directed or not
    """

    # Create graph from sparse matrix and generate its layout (node positions)
    if directed:
        graph = nx.from_scipy_sparse_matrix(adj_mat, create_using=nx.DiGraph())
    else:
        graph = nx.from_scipy_sparse_matrix(adj_mat)
    layout = nx.spring_layout(graph)

    # Add graph nodes
    nx.draw_networkx_nodes(graph, pos=layout, node_size=10)

    if weighted:

        # Obtain all node weights
        edge_weights = list()
        for _, _, data in graph.edges(data=True):
            edge_weights.append(data['weight'])

        # Draw weighted graph edges
        nx.draw_networkx_edges(graph, pos=layout,
                               width=edge_weights,
                               arrows=directed,
                               arrowsize=10,
                               arrowstyle='-|>')

        # Show the graph
        plt.show()
        plt.title('Weighted Graph with Spring Layout')
        plt.savefig("./graph_"+str(idx)+"_label_"+str(label)+".png")

        return

    # Draw undirected graph edges
    nx.draw_networkx_edges(graph,
                           pos=layout,
                           arrows=directed,
                           arrowsize=10,
                           arrowstyle='-|>')

    # Show the graph
    plt.title('Graph with Spring Layout')
    plt.show()


def collate(samples):
    """
    Collate function used by the data loader to put graphs into a batch
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.reshape(torch.tensor(labels, dtype=torch.float), [-1, 1])


class ProstateCancerDataset(Dataset):
    """
    Dataset class for the prostate cancer dataset
    """

    def __init__(self, mat_file_path,
                 fft_mat_file_path=None,
                 train=True,
                 weighted=False,
                 k=10,
                 knn_n_jobs=1,
                 threshold=None,
                 perform_pca=False,
                 num_pca_components=None):
        """
        Constructor for the prostate cancer dataset class
        Parameters:
            mat_file_path (str): Path to the time domain .mat file
            fft_mat_file_path (str): Path to the frequency domain .mat file
            train (bool): Indicates whether train or test data is loaded
            weighted (bool): Indicates whether created graph is weighted or not
            k (int): Number of neighbours to use for the K-nearest neighbour algorithm
            knn_n_jobs (int): Number of jobs to deploy for graph creation
            threshold (float): Value indicating the cutoff value for the Euclidean distance in graph creation (Only
                               valid when weighted is set to True)
            perform_pca (bool): Indicates whether PCA dimension reduction is performed on data or not
            num_pca_components (int): Indicates the number of components for PCA
        """

        # Load the .mat file
        self.prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

        # Load either the test or train data
        self.mat_data = self.prostate_cancer_mat_data['data_train' if train else 'data_test']

        self.use_fft_data = False

        # Use frequency domain data if provided
        if fft_mat_file_path:
            self.use_fft_data = True
            self.prostate_cancer_fft_mat_data = h5py.File(fft_mat_file_path, 'r')
            self.mat_fft_data = self.prostate_cancer_fft_mat_data['data_train' if train else 'data_test']

        # Find the number of available cores
        self.num_cores = self.mat_data.shape[0]

        # Obtain the labels for the cores
        self.labels = np.array(self.prostate_cancer_mat_data['label_train' if train else 'label_test'], dtype=np.int)

        # Parameters used in graph creation
        self.weighted = weighted
        self.k = k
        self.knn_n_jobs = knn_n_jobs
        self.threshold = threshold

        # Other variables
        self.train = train
        self.perform_pca = perform_pca
        self.num_pca_components = num_pca_components

    def __getitem__(self, idx):
        """
        Item iterator for the prostate cancer dataset
        Parameters:
            idx (int): index of data point to retrieve
        Returns:
            (numpy array): Numpy array containing the signals for a single core
            (int): Label indicating whether the core is cancerous or healthy
        """

        # Obtain the label for the specified core
        label = self.labels[idx][0]

        # Obtain the core signals and change them into a numpy array
        data = np.array(self.prostate_cancer_mat_data[self.mat_data[idx, 0]][()].transpose(), dtype=np.float32)

        # Perform PCA on time domain data (To be used as node features)
        if self.perform_pca:
            pca = PCA(n_components=self.num_pca_components)
            pca.fit(data)
            reduced_data = pca.transform(data)

        # Create the graph using the FFT data
        if self.use_fft_data:
            # Use the second half of each FFT signal
            freq_data = np.array(self.prostate_cancer_fft_mat_data[self.mat_fft_data[idx*2+1, 0]][()].transpose())
            freq_data = np.sqrt(np.power(freq_data['real'], 2) + np.power(freq_data['imag'], 2), dtype=np.float32)

            # Perform PCA on FFT data
            if self.perform_pca:
                pca = PCA(n_components=self.num_pca_components)
                pca.fit(freq_data)
                reduced_freq_data = pca.transform(freq_data)

                # Create the graph using reduced FFT data
                graph = create_knn_adj_mat(reduced_freq_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold)
            else:
                # Create the graph using FFT data
                graph = create_knn_adj_mat(freq_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold)
        else:
            if self.perform_pca:
                # noinspection PyUnboundLocalVariable
                graph = create_knn_adj_mat(reduced_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold)
            else:
                graph = create_knn_adj_mat(data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold)

        # Create a dgl graph from coo_matrix
        g = dgl.DGLGraph()
        g.from_scipy_sparse_matrix(graph)

        # Put time domain signals as node features
        if self.perform_pca:
            g.nodes[:].data['x'] = reduced_data
        else:
            g.nodes[:].data['x'] = data

        return g, label

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cores
        """
        return self.num_cores
