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
from datetime import datetime
import logging
import random
from sklearn.manifold import TSNE
try:
    from knn_cuda import KNN
except ImportError:
    pass

logger = logging.getLogger('gnn_prostate_cancer')


def random_distance_extractor(k,
                              input_mat_file,
                              data_label='data_train',
                              n_jobs=5,
                              rows_per_matrix=10,
                              num_cores=100,
                              output_file_path="./dist.txt",
                              fft=False,
                              pca_components=None):
    """
    Create an adjacency matrix with distances and save the distances inside a text file.
    Number of saved values is calculated as: (k-1)*rows_per_matrix
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        n_jobs (int): number of jobs to deploy if GPU is not used
        algorithm (str): Choose between auto, ball_tree, kd_tree or brute
        rows_per_matrix (int): indicates how many rows are randomly chosen to get their distance values
        num_cores (int): Number of cores to extract the distances from
        data_label (str): String indicating what the data is labelled as in the mat file
        fft (bool): Indicates whether the fft mat file is inputted
        pca_components (int): Indicates the number of components in the PCA reduction (no pca if not set)
    """

    # Load the .mat file
    prostate_cancer_mat_data = h5py.File(input_mat_file, 'r')
    mat_data = prostate_cancer_mat_data[data_label]

    # Randomly choose cores
    n_cores = mat_data.shape[0]
    random_core_idx = random.sample(range(0, n_cores), num_cores)

    for idx in random_core_idx:

        if fft:
            # Use the second half of each FFT signal
            data = np.array(prostate_cancer_mat_data[mat_data[idx * 2 + 1, 0]][()].transpose()[:, 0:100:2])
            data = np.sqrt(np.power(data['real'], 2) + np.power(data['imag'], 2), dtype=np.float32)
        else:
            # Obtain the core signals and change them into a numpy array
            data = np.array(prostate_cancer_mat_data[mat_data[idx, 0]][()].transpose(), dtype=np.float32)

        # Perform PCA reduction
        if pca_components:
            pca = PCA(n_components=pca_components)
            pca.fit(data)
            data = pca.transform(data)

        # initialize and fit nearest neighbour algorithm
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, algorithm='auto')
        neigh.fit(data)

        # Obtain matrix with distance of k-nearest points
        adj_mat_weighted = np.array(sp.coo_matrix(neigh.kneighbors_graph(data,
                                                                         k,
                                                                         mode='distance')).toarray())

        # Choose random rows per core
        random_rows = random.sample(range(0, adj_mat_weighted.shape[0]), rows_per_matrix)

        # Extract the distance
        dist = adj_mat_weighted[random_rows][np.nonzero(adj_mat_weighted[random_rows])].flatten()

        # Save to file
        f = open(output_file_path, "a")
        for ele in dist:
            f.write(str(ele) + "\n")
        f.close()


def plot_graph_embeddings(path_to_file, num_itr, path_to_output, figure_title, cg=True, show_cg_on_figure=False):

    """
    Performs TSNE on input graph embeddings and plots them (saves the figure)
    Parameters:
        path_to_file (str): path to text file containing graph embeddings (the last letter in the name should correspond
        to the iteration) (example: "D/embeddings_itr_")
        num_itr (int): number of iterations
        path_to_output (str): the output path to save the figures to
        cg (bool): Indicates whether cancer grade should be shown in the figures
        figure_title (str): figure title (itr number will be appended to this)
        show_cg_on_figure (bool): Indicates whether each point is annotated with its cg label (only a legend otherwise)
    """

    for itr in range(num_itr):

        x = np.genfromtxt(path_to_file+str(itr)+".txt",
                          delimiter=' ',
                          dtype=np.float)

        # Extract labels and embeddings
        if cg:
            labels = x[:, 0]
            x = x[:, 2:]

            cg_labels = np.genfromtxt(path_to_file + str(itr) + ".txt",
                                      delimiter=' ',
                                      dtype=str)[:, 1]

            x = np.delete(x, 0, axis=1)
        else:
            labels = x[:, 0]
            x = x[:, 1:]

        # Perform TSNE reduction
        x_embedded = TSNE(n_components=2, random_state=5).fit_transform(x)

        if cg:

            # Colours dictionary
            cdict = {"['-']": "green",
                     "['1+1']": "green",
                     "['1+2']": "green",
                     "['2+1']": "green",
                     "['2+2']": "green",
                     "['2+3']": "green",
                     "['3+2']": "green",
                     "['3+3']": "pink",
                     "['3+4']": "orange",
                     "['4+3']": "purple",
                     "['4+4']": "olive",
                     "['4+5']": "red",
                     "['5+4']": "blue",
                     "['5+5']": "cyan"
                     }

            fig, ax = plt.subplots()

            for cgl in np.unique(cg_labels):
                color = cdict[cgl]
                indices = np.where(cg_labels == cgl)[0]
                ax.scatter(x_embedded[indices, 0], x_embedded[indices, 1], c=color, label=cgl)

            # Show CG labels for each scatter point
            if show_cg_on_figure:
                for j, (x, y) in enumerate(zip(x_embedded[:, 0], x_embedded[:, 1])):
                    if labels[j] == 1:
                        plt.annotate(cg_labels[j],  # this is the text
                                     (x, y),  # this is the point to label
                                     textcoords="offset points",  # how to position the text
                                     xytext=(0, 10),  # distance from text to points (x,y)
                                     ha='center')  # horizontal alignment can be left, right or center

            ax.legend()
            ax.grid(True)

        else:
            color = ['green' if l == 0 else 'red' for l in labels]

            # Draw the scatter plot with the created colors
            plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=color)

        plt.title( figure_title+str(itr))
        plt.savefig(path_to_output+str(itr))
        plt.close()


def create_knn_adj_mat(features, k, weighted=False, n_jobs=None, algorithm='auto', threshold=None, use_gpu=False):
    """
    Create a directed normalized adjacency matrix from input nodes based on k-nearest neighbours
    Parameters:
        features (numpy array of node features): array of size N*M (N nodes, M feature size)
        k (int): number of neighbours to find
        weighted (bool): set to True for weighted adjacency matrix (based on Euclidean distance)
        n_jobs (int): number of jobs to deploy if GPU is not used
        algorithm (str): Choose between auto, ball_tree, kd_tree or brute
        threshold (float): Cutoff value for the Euclidean distance
        use_gpu (bool): Indicates whether GPU is to be used for the KNN algorithm
    Returns:
        (coo matrix): adjacency matrix as a sparse coo matrix
    """
    t_start = datetime.now()

    if use_gpu:

        features_extra_dim = np.expand_dims(features, axis=2)

        knn = KNN(k=k, transpose_mode=True)

        # Find the k nearest neighbours and their distance
        dist, idx = knn(torch.from_numpy(features_extra_dim).cuda(), torch.from_numpy(features_extra_dim).clone().cuda())

        torch.cuda.empty_cache()

        del features_extra_dim

        idx = idx.cpu()
        dist = dist.cpu()
        
        # Clean up the indices and distances
        dist = dist.flatten()

        # Create tuples of indices where an edge exists
        rows = np.repeat(np.arange(features.shape[0]), k)
        columns = idx.flatten()
        non_zero_indices = tuple(np.stack((rows, columns)))

        del rows
        del columns
        del idx

        # Remove edges where the distance is higher than the threshold
        if threshold:
            indices_to_remove = dist > threshold
            indices_to_remove = np.where(indices_to_remove)
            non_zero_indices = tuple(np.delete(non_zero_indices, indices_to_remove, 1))
            dist = np.delete(dist, indices_to_remove[0], 0)

            del indices_to_remove

        if weighted:

            # Create zero matrix as the initial adjacency matrix
            adj_mat_weighted = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)

            # Fill in the adjacency matrix with node distances
            adj_mat_weighted[non_zero_indices] = dist

            non_zero_indices = np.nonzero(adj_mat_weighted)

            # Take reciprocal of non-zero elements to associate lower weight to higher distances
            adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

            # Normalize rows
            coo_matrix = sp.coo_matrix(adj_mat_weighted)
            normalized_coo_matrix = normalize(coo_matrix)

            # DGL requires self loops
            normalized_coo_matrix = normalized_coo_matrix + sp.eye(normalized_coo_matrix.shape[0])

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return normalized_coo_matrix

        else:
            # Create eye matrix as the initial adjacency matrix
            adj_mat_binary = np.zeros((features.shape[0], features.shape[0]))

            # Create the binary adjacency matrix
            adj_mat_binary[non_zero_indices] = 1

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return sp.coo_matrix(adj_mat_binary)
    else:

        # initialize and fit nearest neighbour algorithm
        neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, algorithm=algorithm)
        neigh.fit(features)

        # Obtain matrix with distance of k-nearest points
        adj_mat_weighted = np.array(sp.coo_matrix(neigh.kneighbors_graph(features,
                                                                         k,
                                                                         mode='distance')).toarray())

        if threshold:
            indices_to_zero = adj_mat_weighted > threshold
            adj_mat_weighted[indices_to_zero] = 0

        non_zero_indices = np.nonzero(adj_mat_weighted)

        if weighted:

            # Take reciprocal of non-zero elements to associate lower weight to higher distances
            adj_mat_weighted[non_zero_indices] = 1 / adj_mat_weighted[non_zero_indices]

            # Normalize rows
            adj_mat_weighted = sp.coo_matrix(adj_mat_weighted)
            normalized_coo_matrix = normalize(adj_mat_weighted)

            # DGL requires self loops
            normalized_coo_matrix = normalized_coo_matrix + sp.eye(normalized_coo_matrix.shape[0])

            del adj_mat_weighted

            t_end = datetime.now()
            logger.debug("it took {} to create the graph".format(t_end - t_start))

            return normalized_coo_matrix

        # Obtain the binary adjacency matrix
        adj_mat_binary = adj_mat_weighted
        adj_mat_binary[non_zero_indices] = 1
        adj_mat_binary = adj_mat_binary + np.eye(adj_mat_binary.shape[0])

        del adj_mat_weighted

        t_end = datetime.now()
        logger.debug("it took {} to create the graph".format(t_end - t_start))

        return sp.coo_matrix(adj_mat_binary)


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
    graphs, labels, cg = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.reshape(torch.tensor(labels, dtype=torch.float), [-1, 1]), cg


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
                 cuda_knn=False,
                 threshold=None,
                 perform_pca=False,
                 num_pca_components=None,
                 get_cancer_grade=False,
                 cancer_grade_string='GS_train',
                 train_data_string='data_train',
                 test_data_string='data_test',
                 train_fft_data_string='data_train',
                 test_fft_data_string='data_test',
                 train_data_label_string='label_train',
                 test_data_label_string='label_test'
                 ):
        """
        Constructor for the prostate cancer dataset class
        Parameters:
            mat_file_path (str): Path to the time domain .mat file
            fft_mat_file_path (str): Path to the frequency domain .mat file
            train (bool): Indicates whether train or test data is loaded
            weighted (bool): Indicates whether created graph is weighted or not
            k (int): Number of neighbours to use for the K-nearest neighbour algorithm
            knn_n_jobs (int): Number of jobs to deploy for graph creation in CPU mode
            cuda_knn (bool): Indicate whether GPU knn is used or not
            threshold (float): Value indicating the cutoff value for the Euclidean distance in graph creation (Only
                               valid when weighted is set to True)
            perform_pca (bool): Indicates whether PCA dimension reduction is performed on data or not
            num_pca_components (int): Indicates the number of components for PCA
            get_cancer_grade (bool): Indicates whether cancer grade is to be extracted from the mat file
            train_data_string (str): String associated with cancer grade in the mat file
            train_data_string (str): If train data string is anything other than data_train, specify it
            test_data_string (str): If test data string is anything other than data_test, specify it
            train_fft_data_string (str): If train data string is anything other than data_train, specify it
            test_fft_data_string (str): If test data string is anything other than data_test, specify it
            train_data_label_string (str): If train data string is anything other than data_train, specify it
            test_data_label_string (str): If test data string is anything other than data_test, specify it
        """

        # Load the .mat file
        self.prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

        # Load either the test or train data
        self.mat_data = self.prostate_cancer_mat_data[train_data_string if train else test_data_string]

        self.use_fft_data = False

        # Use frequency domain data if provided
        if fft_mat_file_path:
            self.use_fft_data = True
            self.prostate_cancer_fft_mat_data = h5py.File(fft_mat_file_path, 'r')
            self.mat_fft_data = self.prostate_cancer_fft_mat_data[
                train_fft_data_string if train else test_fft_data_string]

        # Find the number of available cores
        self.num_cores = self.mat_data.shape[0]

        # Obtain the labels for the cores
        self.labels = np.array(self.prostate_cancer_mat_data[
                                   train_data_label_string if train else test_data_label_string], dtype=np.int)

        if get_cancer_grade:
            self.cg = self.prostate_cancer_mat_data[cancer_grade_string]

        # Parameters used in graph creation
        self.weighted = weighted
        self.k = k
        self.knn_n_jobs = knn_n_jobs
        self.threshold = threshold
        self.cuda_knn = cuda_knn

        # Other variables
        self.train = train
        self.perform_pca = perform_pca
        self.num_pca_components = num_pca_components
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_cancer_grade = get_cancer_grade

    def __getitem__(self, idx):
        """
        Item iterator for the prostate cancer dataset
        Parameters:
            idx (int): index of data point to retrieve
        Returns:
            (numpy array): Numpy array containing the signals for a single core
            (int): Label indicating whether the core is cancerous or healthy
        """
        if idx is 45:
            idx = 46

        # Obtain the label for the specified core
        label = self.labels[idx][0]

        # Obtain the cancer grade
        cg = None
        if self.get_cancer_grade:
            cg = self.prostate_cancer_mat_data[self.cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)

        # Obtain the core signals and change them into a numpy array
        data = np.array(self.prostate_cancer_mat_data[self.mat_data[idx, 0]][()].transpose(), dtype=np.float32)

        # Perform PCA on time domain data (To be used as node features)
        if self.perform_pca:
            t_start = datetime.now()
            pca = PCA(n_components=self.num_pca_components)
            pca.fit(data)
            reduced_data = pca.transform(data)
            t_end = datetime.now()
            logger.debug("PCA Reduction for RF data took {}".format(t_end - t_start))

        # Create the graph using the FFT data
        if self.use_fft_data:
            # Use the second half of each FFT signal
            freq_data = np.array(self.prostate_cancer_fft_mat_data[self.mat_fft_data[idx*2+1, 0]][()].transpose())
            freq_data = np.sqrt(np.power(freq_data['real'], 2) + np.power(freq_data['imag'], 2), dtype=np.float32)

            # Perform PCA on FFT data
            if self.perform_pca:
                t_start = datetime.now()
                pca = PCA(n_components=self.num_pca_components)
                pca.fit(freq_data)
                reduced_freq_data = pca.transform(freq_data)
                t_end = datetime.now()
                logger.debug("PCA Reduction for FFT data took {}".format(t_end - t_start))

                # Create the graph using reduced FFT data
                graph = create_knn_adj_mat(reduced_freq_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold,
                                           use_gpu=self.cuda_knn)
            else:
                # Create the graph using FFT data
                graph = create_knn_adj_mat(freq_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold,
                                           use_gpu=self.cuda_knn)
        else:
            if self.perform_pca:
                # noinspection PyUnboundLocalVariable
                graph = create_knn_adj_mat(reduced_data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold,
                                           use_gpu=self.cuda_knn)
            else:
                graph = create_knn_adj_mat(data,
                                           k=self.k,
                                           weighted=self.weighted,
                                           n_jobs=self.knn_n_jobs,
                                           threshold=self.threshold,
                                           use_gpu=self.cuda_knn)

        # Create a dgl graph from coo_matrix
        g = dgl.from_scipy(graph, device=self.device)

        # Put time domain signals as node features
        if self.perform_pca:
            g.nodes[:].data['x'] = torch.from_numpy(reduced_data).to(self.device)
        else:
            g.nodes[:].data['x'] = torch.from_numpy(data).to(self.device)

        return g, label, cg

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cores
        """
        return self.num_cores


def main():
    print("this is main")

    random_distance_extractor(k=40,
                              input_mat_file="D:/Workplace/ML/Repositories/prostate_cancer_gnn/data/BK_RF_P1_90_MICCAI_33.mat",
                              data_label='data_train',
                              n_jobs=5,
                              rows_per_matrix=10,
                              num_cores=100,
                              output_file_path='D:/Workplace/ML/Documents/distance.txt',
                              fft=False,
                              pca_components=50)

    plot_graph_embeddings(path_to_file="D:/Workplace/ML/Documents/graphsage_12-4train_graph_embeddings_itr_",
                          num_itr=100,
                          path_to_output="./graphsage_12-4_itr_",
                          figure_title="graphsage_12-4 -- itr ",
                          cg=True,
                          show_cg_on_figure=False)

if __name__ == "__main__":
    main()
