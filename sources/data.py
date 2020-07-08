from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import torch
import dgl


def create_knn_adj_mat(features, k, weighted=False, n_jobs=None, algorithm='auto'):
    """
    Create a directed normalized adjacency matrix from input nodes based on k-nearest neighbours
    Parameters:
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        weighted (bool): set to True for weighted adjacency matrix (based on Euclidean distance)
        n_jobs (int): number of jobs to deploy
        algorithm (str): Choose between auto, ball_tree, kd_tree or brute
    Returns:
        (coo matrix): adjacency matrix as a sparse coo matrix
    """

    # initialize and fit nearest neighbour algorithm
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, algorithm='ball_tree')
    neigh.fit(features)

    # Obtain the binary adjacency matrix
    adj_mat_connectivity = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='connectivity')).toarray())

    # Remove self-connections
    adj_mat_connectivity -= np.eye(features.shape[0])

    if weighted:
        # Obtain matrix with distance of k-nearest points
        adj_mat_weighted = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='distance')).toarray())

        # Take reciprocal of non-zero elements to associate lower weight to higher distances
        non_zero_indices = np.nonzero(adj_mat_connectivity)
        adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

        # Normalize rows
        adj_mat_weighted = adj_mat_weighted / adj_mat_weighted.sum(1)[:, np.newaxis]

        return sp.coo_matrix(adj_mat_weighted)

    return sp.coo_matrix(adj_mat_connectivity)


def draw_graph(adj_mat, weighted=False, directed=False):
    """
    Plots the graph for a weighted adjacency matrix
    Parameters:
        adj_mat (sparse coo matrix): Weighted adjacency matrix
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
        plt.title('Weighted Graph with Spring Layout')
        plt.show()

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

    def __init__(self, mat_file_path, train=True, weighted=False, k=10, n_jobs=1, knn_algorithm='auto'):
        """
        Constructor for the prostate cancer dataset class
        Parameters:
            mat_file_path (str): Path to the .mat file
            train (bool): Indicates whether train or test data is loaded
            weighted (bool): Indicates whether created graph is weighted or not
            k (int): Number of neighbours to use for the K-nearest neighbour algorithm
            n_jobs (int): Number of jobs to deploy for graph creation
            knn_algorithm (str): Choose between auto, ball_tree, kd_tree or brute for the graph creation algorithm
        """

        # Load the .mat file
        self.prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

        # Load either the test or train data
        self.mat_data = self.prostate_cancer_mat_data['data_train' if train else 'data_test']

        # Find the number of available cells
        self.num_cells = self.mat_data.shape[0]

        # Obtain the labels for the cells
        self.labels = np.array(self.prostate_cancer_mat_data['label_train' if train else 'label_test'], dtype=np.int)

        # Parameters used in graph creation
        self.weighted = weighted
        self.k = k
        self.n_jobs = n_jobs
        self.knn_algorithm = knn_algorithm

    def __getitem__(self, idx):
        """
        Item iterator for the prostate cancer dataset
        Parameters:
            idx (int): index of data point to retrieve
        Returns:
            (numpy array): Numpy array containing the signals for a single cell
            (int): Label indicating whether the cell is cancerous or healthy
        """

        # Obtain the label for the specified cell
        label = self.labels[idx][0]

        # Obtain the cell signals and change them into a numpy array
        data = np.array(self.prostate_cancer_mat_data[self.mat_data[idx, 0]][()].transpose(), dtype=np.float32)

        # Create the graph using knn
        graph = create_knn_adj_mat(data,
                                   k=self.k,
                                   weighted=self.weighted,
                                   n_jobs=self.n_jobs,
                                   algorithm=self.knn_algorithm)

        # Create a dgl graph from coo_matrix
        g = dgl.DGLGraph()
        g.from_scipy_sparse_matrix(graph)

        # Put RF signals as node features
        g.nodes[:].data['x'] = data

        return g, label

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cells
        """
        return self.num_cells
