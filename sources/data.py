from sklearn.neighbors import NearestNeighbors
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt


def create_knn_adj_mat(features, k, weighted=False, n_jobs=None):
    """
    Create a directed normalized adjacency matrix from input nodes based on k-nearest neighbours
    Parameters:
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        weighted (bool): set to True for weighted adjacency matrix (based on Euclidean distance)
        n_jobs (int): number of jobs to deploy
    Returns:
        (coo matrix): adjacency matrix as a sparse coo matrix
    """

    # initialize and fit nearest neighbour algorithm
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
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
