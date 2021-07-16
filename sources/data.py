from numpy import core
from torch._C import dtype
from data_utils import create_knn_adj_mat
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import h5py
import torch
import dgl
from sklearn.decomposition import PCA
from datetime import datetime
import logging
from numpy.random import choice
try:
    from knn_cuda import KNN
except ImportError:
    pass


logger = logging.getLogger('gnn_prostate_cancer')


def node_classification_knn_graph(mat_file_path,
                                  weighted=False,
                                  k=10,
                                  knn_n_jobs=1,
                                  cuda_knn=False,
                                  threshold=None,
                                  perform_pca=False,
                                  num_pca_components=50,
                                  get_cancer_grade=False,
                                  train_data_string='data_train',
                                  test_data_string='data_test',
                                  val_data_string='data_val',
                                  train_label_string='label_train',
                                  test_label_string='label_test',
                                  val_label_string='label_val',
                                  train_cancer_grade_string='GS_train',
                                  val_cancer_grade_string='GS_val',
                                  test_cancer_grade_string='GS_test',
                                  num_signals=1500,
                                  signal_level_graph=False):
    """
    Creates the knn graph containing training, validation and test nodes
    Parameters:
        mat_file_path (str): Path to the time series .mat file
        weighted (bool): Indicates whether the constructed graph is weighted or not
        k (int): Number of neighbours to use for the K-nearest neighbour algorithm
        knn_n_jobs (int): Number of jobs to deploy for graph creation in CPU mode
        cuda_knn (bool): Indicate whether GPU knn is used or not
        threshold (float): Value indicating the cutoff value for the Euclidean distance in graph creation (Only
                           valid when weighted is set to True)
        perform_pca (bool): Indicates whether PCA dimension reduction is performed on data or not
        num_pca_components (int): Indicates the number of components for PCA
        get_cancer_grade (bool): Indicates whether cancer grade labels are obtained or not
        train_data_string (str): If train data string is anything other than data_train, specify it
        test_data_string (str): If test data string is anything other than data_test, specify it
        val_data_string (str): If validation data string is anything other than data_val, specify it
        train_label_string (str): If train data string is anything other than data_train, specify it
        test_label_string (str): If test data string is anything other than data_test, specify it
        val_label_string (str): If validation data string is anything other than data_val, specify it
        train_cancer_grade_string (str): Cancer grade string in mat file
        val_cancer_grade_string (str): Cancer grade string in mat file
        test_cancer_grade_string (str): Cancer grade string in mat file
        core_location_graph (bool): Indicates whether the core location graph (no knn) is constructed
        num_signals (int): Number of signals to use per core
        train_corename_string (str): Training corename string in the mat file
        val_corename_string (str): Validation corename string in the mat file
        test_corename_string (str): Test corename string in the mat file
        signal_level_graph (bool): Indicates whether the KNN graph is built for
                                   cores (mean of signals) or signals
    """
    # Load the .mat file
    prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

    # Load either the test or train data
    train_mat_data = prostate_cancer_mat_data[train_data_string]
    val_mat_data = prostate_cancer_mat_data[val_data_string]
    test_mat_data = prostate_cancer_mat_data[test_data_string]

    # Load the labels
    train_labels = np.array(prostate_cancer_mat_data[train_label_string], dtype=np.int8)
    val_labels = np.array(prostate_cancer_mat_data[val_label_string], dtype=np.int8)
    test_labels = np.array(prostate_cancer_mat_data[test_label_string], dtype=np.int8)
    labels = np.concatenate((train_labels, val_labels, test_labels)).reshape((-1, 1))
    if signal_level_graph:
        labels = np.repeat(labels, num_signals)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Get total number of cores (nodes)
    train_cores_num = train_mat_data.shape[0]
    val_cores_num = val_mat_data.shape[0]
    test_cores_num = test_mat_data.shape[0]
    num_cores = train_cores_num + val_cores_num + test_cores_num

    # Get the number of nodes
    if signal_level_graph:
        train_nodes_num = train_cores_num*num_signals
        val_nodes_num = val_cores_num*num_signals
        test_nodes_num = test_cores_num*num_signals
        num_nodes = num_cores*num_signals
    else:
        train_nodes_num = train_cores_num
        val_nodes_num = val_cores_num
        test_nodes_num = test_cores_num
        num_nodes = num_cores

    # Get cancer grade labels
    cgs = None
    if get_cancer_grade:

        train_cgs = list()
        val_cgs = list()
        test_cgs = list()

        for idx in range(train_cores_num):
            cg = prostate_cancer_mat_data[train_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            train_cgs.append(cg)

        for idx in range(val_cores_num):
            cg = prostate_cancer_mat_data[val_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            val_cgs.append(cg)

        for idx in range(test_cores_num):
            cg = prostate_cancer_mat_data[test_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            test_cgs.append(cg)

        cgs = np.concatenate((train_cgs, val_cgs, test_cgs))
        if signal_level_graph:
            cgs = np.repeat(cgs, num_signals)

    # Determine the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create train/test/val masks
    trainmsk = np.zeros(num_nodes, dtype=np.int8)
    valmsk = np.zeros(num_nodes, dtype=np.int8)
    testmsk = np.zeros(num_nodes, dtype=np.int8)
    trainmsk[range(train_nodes_num)] = 1
    valmsk[range(train_nodes_num, train_nodes_num + val_nodes_num)] = 1
    testmsk[range(train_nodes_num + val_nodes_num, train_nodes_num + val_nodes_num + test_nodes_num)] = 1
    trainmsk = torch.tensor(trainmsk > 0)
    valmsk = torch.tensor(valmsk > 0)
    testmsk = torch.tensor(testmsk > 0)
    mask = {'Training': trainmsk, 'Validation': valmsk, 'Test': testmsk}

    # Get sample data
    cores_data = list()
    if signal_level_graph:
        for idx in range(train_cores_num):
            indices = np.sort(choice(a=range(prostate_cancer_mat_data[train_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
            cores_data.append(np.expand_dims(
                np.swapaxes(prostate_cancer_mat_data[train_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
        for idx in range(val_cores_num):
            indices = np.sort(choice(a=range(prostate_cancer_mat_data[val_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
            cores_data.append(np.expand_dims(
                np.swapaxes(prostate_cancer_mat_data[val_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
        for idx in range(test_cores_num):
            indices = np.sort(choice(a=range(prostate_cancer_mat_data[test_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
            cores_data.append(np.expand_dims(
                np.swapaxes(prostate_cancer_mat_data[test_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
        cores_data = np.vstack(cores_data).astype(np.float32)
        cores_data = np.reshape(cores_data, (num_cores*num_signals, -1))
    else:
        for idx in range(train_cores_num):
            cores_data.append(np.mean(prostate_cancer_mat_data[train_mat_data[idx, 0]][()], axis=1))
        for idx in range(val_cores_num):
            cores_data.append(np.mean(prostate_cancer_mat_data[val_mat_data[idx, 0]][()], axis=1))
        for idx in range(test_cores_num):
            cores_data.append(np.mean(prostate_cancer_mat_data[test_mat_data[idx, 0]][()], axis=1))
        cores_data = np.array(cores_data, dtype=np.float32)

    # Perform PCA on time domain data (To be used as node features)
    if perform_pca:
        t_start = datetime.now()
        pca = PCA(n_components=num_pca_components)
        pca.fit(cores_data)
        cores_data = pca.transform(cores_data)
        t_end = datetime.now()
        logger.debug("PCA Reduction for RF data took {}".format(t_end - t_start))

    graph = create_knn_adj_mat(cores_data,
                               k=k,
                               weighted=weighted,
                               n_jobs=knn_n_jobs,
                               threshold=threshold,
                               use_gpu=cuda_knn)

    # Create a dgl graph from coo_matrix
    g = dgl.from_scipy(graph)
    g = dgl.add_self_loop(g)

    # Move graph to available device
    g = g.to(device)
    labels = labels.to(device)

    # Put time domain signals as node features
    g.ndata['h'] = torch.from_numpy(cores_data).to(device)

    return g, labels, mask, cgs


def node_classification_core_location_graph(mat_file_path,
                                            perform_pca=False,
                                            num_pca_components=50,
                                            get_cancer_grade=False,
                                            train_data_string='data_train',
                                            test_data_string='data_test',
                                            val_data_string='data_val',
                                            train_label_string='label_train',
                                            test_label_string='label_test',
                                            val_label_string='label_val',
                                            train_cancer_grade_string='GS_train',
                                            val_cancer_grade_string='GS_val',
                                            test_cancer_grade_string='GS_test',
                                            num_signals=1500,
                                            train_corename_string='corename_train',
                                            val_corename_string='corename_val',
                                            test_corename_string='corename_test',
                                            signal_level_graph=False):
    """
    Creates the knn graph containing training, validation and test nodes
    Parameters:
        mat_file_path (str): Path to the time series .mat file
        weighted (bool): Indicates whether the constructed graph is weighted or not
        k (int): Number of neighbours to use for the K-nearest neighbour algorithm
        knn_n_jobs (int): Number of jobs to deploy for graph creation in CPU mode
        cuda_knn (bool): Indicate whether GPU knn is used or not
        threshold (float): Value indicating the cutoff value for the Euclidean distance in graph creation (Only
                           valid when weighted is set to True)
        perform_pca (bool): Indicates whether PCA dimension reduction is performed on data or not
        num_pca_components (int): Indicates the number of components for PCA
        get_cancer_grade (bool): Indicates whether cancer grade labels are obtained or not
        train_data_string (str): If train data string is anything other than data_train, specify it
        test_data_string (str): If test data string is anything other than data_test, specify it
        val_data_string (str): If validation data string is anything other than data_val, specify it
        train_label_string (str): If train data string is anything other than data_train, specify it
        test_label_string (str): If test data string is anything other than data_test, specify it
        val_label_string (str): If validation data string is anything other than data_val, specify it
        train_cancer_grade_string (str): Cancer grade string in mat file
        val_cancer_grade_string (str): Cancer grade string in mat file
        test_cancer_grade_string (str): Cancer grade string in mat file
        num_signals (int): Number of signals to use per core
        train_corename_string (str): Training corename string in the mat file
        val_corename_string (str): Validation corename string in the mat file
        test_corename_string (str): Test corename string in the mat file
        signal_level_graph (bool): Indicates whether the KNN graph is built for 
                                   cores (mean of signals) or signals
    """
    # Load the .mat file
    prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

    # Load either the test or train data
    train_mat_data = prostate_cancer_mat_data[train_data_string]
    val_mat_data = prostate_cancer_mat_data[val_data_string]
    test_mat_data = prostate_cancer_mat_data[test_data_string]

    # Load the labels
    train_labels = np.array(prostate_cancer_mat_data[train_label_string], dtype=np.int8)
    val_labels = np.array(prostate_cancer_mat_data[val_label_string], dtype=np.int8)
    test_labels = np.array(prostate_cancer_mat_data[test_label_string], dtype=np.int8)
    labels = np.concatenate((train_labels, val_labels, test_labels)).reshape((-1, 1))
    if signal_level_graph:
        labels = np.repeat(labels, num_signals)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Get total number of cores
    train_cores_num = train_mat_data.shape[0]
    val_cores_num = val_mat_data.shape[0]
    test_cores_num = test_mat_data.shape[0]
    num_cores = train_cores_num + val_cores_num + test_cores_num

    # Get the number of nodes
    if signal_level_graph:
        train_nodes_num = train_cores_num*num_signals
        val_nodes_num = val_cores_num*num_signals
        test_nodes_num = test_cores_num*num_signals
        num_nodes = num_cores*num_signals
    else:
        train_nodes_num = train_cores_num
        val_nodes_num = val_cores_num
        test_nodes_num = test_cores_num
        num_nodes = num_cores

    # Get cancer grade labels
    cgs = None
    if get_cancer_grade:

        train_cgs = list()
        val_cgs = list()
        test_cgs = list()

        for idx in range(train_cores_num):
            cg = prostate_cancer_mat_data[train_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            train_cgs.append(cg)

        for idx in range(val_cores_num):
            cg = prostate_cancer_mat_data[val_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            val_cgs.append(cg)

        for idx in range(test_cores_num):
            cg = prostate_cancer_mat_data[test_cancer_grade_string]
            cg = prostate_cancer_mat_data[cg[idx][0]]
            cg = ''.join(chr(i) for i in cg)
            test_cgs.append(cg)

        cgs = np.concatenate((train_cgs, val_cgs, test_cgs))
        if signal_level_graph:
            cgs = np.repeat(cgs, num_signals)

    # Determine the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create train/test/val masks
    trainmsk = np.zeros(num_nodes, dtype=np.int8)
    valmsk = np.zeros(num_nodes, dtype=np.int8)
    testmsk = np.zeros(num_nodes, dtype=np.int8)
    trainmsk[range(train_nodes_num)] = 1
    valmsk[range(train_nodes_num, train_nodes_num + val_nodes_num)] = 1
    testmsk[range(train_nodes_num + val_nodes_num, train_nodes_num + val_nodes_num + test_nodes_num)] = 1
    trainmsk = torch.tensor(trainmsk > 0)
    valmsk = torch.tensor(valmsk > 0)
    testmsk = torch.tensor(testmsk > 0)
    mask = {'Training': trainmsk, 'Validation': valmsk, 'Test': testmsk}

    # Get sample data
    cores_data = list()
    for idx in range(train_cores_num):
        indices = np.sort(choice(a=range(prostate_cancer_mat_data[train_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
        cores_data.append(np.expand_dims(
            np.swapaxes(prostate_cancer_mat_data[train_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
    for idx in range(val_cores_num):
        indices = np.sort(choice(a=range(prostate_cancer_mat_data[val_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
        cores_data.append(np.expand_dims(
            np.swapaxes(prostate_cancer_mat_data[val_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
    for idx in range(test_cores_num):
        indices = np.sort(choice(a=range(prostate_cancer_mat_data[test_mat_data[idx, 0]][()].shape[1]), size=num_signals, replace=False))
        cores_data.append(np.expand_dims(
            np.swapaxes(prostate_cancer_mat_data[test_mat_data[idx, 0]][()][:, indices], 0, 1), axis=0))
    cores_data = np.vstack(cores_data).astype(np.float32)
    cores_data = np.reshape(cores_data, (num_cores*num_signals, -1))

    # Perform PCA on time domain data (To be used as node features)
    if perform_pca:
        t_start = datetime.now()
        pca = PCA(n_components=num_pca_components)
        pca.fit(cores_data)
        cores_data = pca.transform(cores_data)
        t_end = datetime.now()
        logger.debug("PCA Reduction for RF data took {}".format(t_end - t_start))

    # Reshape the data samples if needed
    if not signal_level_graph:
        cores_data = np.reshape(cores_data, (num_cores, num_signals, -1))

    # Create core location graph in networkx
    core_locations = np.concatenate((np.array(prostate_cancer_mat_data[train_corename_string], dtype=np.int8),
                                     np.array(prostate_cancer_mat_data[val_corename_string], dtype=np.int8),
                                     np.array(prostate_cancer_mat_data[test_corename_string], dtype=np.int8)), axis=1)

    if signal_level_graph:
        core_locations = np.repeat(core_locations, num_signals, axis=1)

    # Create graph connections based on core location
    graph = list()
    for i in range(8):
        graph.append(nx.complete_graph(np.where(core_locations[i] == 1)[0]))
    graph = nx.compose_all(graph)

    # Create dgl graph
    g = dgl.from_networkx(graph)
    g = g.to(device)
    labels = labels.to(device)

    # Adding node features
    g.ndata['h'] = torch.from_numpy(cores_data).type(torch.float32).to(device)

    return g, labels, mask, cgs


def collate(samples):
    """
    Collate function used by the data loader to put graphs into a batch
    """
    graphs, labels, cg = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.float), cg


class ProstateCancerDataset(Dataset):
    """
    Dataset class for the prostate cancer dataset
    """

    def __init__(self,
                 mat_file_path,
                 mode='train',
                 weighted=False,
                 k=10,
                 knn_n_jobs=1,
                 cuda_knn=False,
                 threshold=None,
                 perform_pca=False,
                 num_pca_components=50,
                 cancer_grade_string=None,
                 train_data_string='data_train',
                 test_data_string='data_test',
                 val_data_string='data_val',
                 train_label_string='label_train',
                 test_label_string='label_test',
                 val_label_string='label_val'):
        """
        Constructor for the prostate cancer dataset class
        Parameters:
            mat_file_path (str): Path to the time series .mat file
            mode (str): Indicates whether 'train', 'test' or 'val' data is loaded
            weighted (bool): Indicates whether created graph is weighted or not
            k (int): Number of neighbours to use for the K-nearest neighbour algorithm
            knn_n_jobs (int): Number of jobs to deploy for graph creation in CPU mode
            cuda_knn (bool): Indicate whether GPU knn is used or not
            threshold (float): Value indicating the cutoff value for the Euclidean distance in graph creation (Only
                               valid when weighted is set to True)
            perform_pca (bool): Indicates whether PCA dimension reduction is performed on data or not
            num_pca_components (int): Indicates the number of components for PCA
            cancer_grade_string (str): String associated with cancer grade in the mat file
            train_data_string (str): If train data string is anything other than data_train, specify it
            test_data_string (str): If test data string is anything other than data_test, specify it
            val_data_string (str): If validation data string is anything other than data_val, specify it
            train_label_string (str): If train data string is anything other than data_train, specify it
            test_label_string (str): If test data string is anything other than data_test, specify it
            val_label_string (str): If validation data string is anything other than data_val, specify it
        """

        # Load the .mat file
        self.prostate_cancer_mat_data = h5py.File(mat_file_path, 'r')

        # Load either the test, train or val data
        assert mode in ['train', 'val', 'test'], 'Invalid mode selected for dataset.'
        if mode == 'train':
            self.mat_data = self.prostate_cancer_mat_data[train_data_string]
            self.labels = np.array(self.prostate_cancer_mat_data[train_label_string], dtype=np.int8)
        elif mode == 'test':
            self.mat_data = self.prostate_cancer_mat_data[test_data_string]
            self.labels = np.array(self.prostate_cancer_mat_data[test_label_string], dtype=np.int8)
        else:
            self.mat_data = self.prostate_cancer_mat_data[val_data_string]
            self.labels = np.array(self.prostate_cancer_mat_data[val_label_string], dtype=np.int8)

        # Find the number of available cores
        self.num_cores = self.mat_data.shape[0]

        # Get the cancer grade
        self.cg_available = False
        if cancer_grade_string:
            self.cg = self.prostate_cancer_mat_data[cancer_grade_string]
            self.cg_available = True

        # Parameters used in graph creation
        self.weighted = weighted
        self.k = k
        self.knn_n_jobs = knn_n_jobs
        self.threshold = threshold
        self.cuda_knn = cuda_knn

        # Other variables
        self.mode = mode
        self.perform_pca = perform_pca
        self.num_pca_components = num_pca_components
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # Obtain the cancer grade
        cg = None
        if self.cg_available:
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
        g = dgl.from_scipy(graph)
        g = dgl.add_self_loop(g)
        g = g.to(self.device)

        # Put time domain signals as node features
        if self.perform_pca:
            g.ndata['h'] = torch.from_numpy(reduced_data).to(self.device)
        else:
            g.ndata['h'] = torch.from_numpy(data).to(self.device)

        return g, label, cg

    def __len__(self):
        """
        Returns:
            (int): Indicates the number of available cores
        """
        return self.num_cores

