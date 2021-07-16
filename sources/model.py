from dgl.nn import GraphConv
from dgl.nn.pytorch import SGConv
from dgl.nn.pytorch import AvgPooling
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import SAGEConv
from torch.nn import Dropout2d, Conv1d
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import dgl
import torch
from math import floor


class NodeBinaryClassifier(nn.Module):
    """
    Node classification model for the prostate cancer dataset
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 aggregator_type='mean',
                 feat_drop=0,
                 use_cuda=False,
                 fc_dropout_p=0,
                 conv_dropout_p=0,
                 attn_drop=0,
                 conv_type='sage',
                 num_heads=3,
                 apply_output_activation=False,
                 conv1d_kernel_size=10,
                 conv1d_stride=8,
                 num_signal_channels=1,
                 signal_level_graph=False,
                 core_location_graph=False):
        """
        Constructor for the NodeBinaryClassifier class
        Parameters:
            input_dim (int): dimension of features for each node
            hidden_dim (int): dimension of hidden embeddings
            aggregator_type (str): One of mean, lstm, gcn or pool
            feat_drop (float): Indicates the dropout rate for the features
            use_cuda (bool): Indicates whether GPU should be utilized or not
            fc_dropout_p (float): Indicates the FC layer dropout ratio
            attn_drop (float): Indicates the dropout rate for the attention mechanism
            conv_type (str): Type of GNN to use
            num_heads (int): Number of attention heads for the GAT network
            apply_output_activation (bool): Indicates whether sigmoid is applied at the output or not
            conv1d_kernel_size (int): Kernel size for 1D conv
            conv1d_stride (int): Stride used for 1D conv
            num_signal_channels (int): Number of signals per core
            signal_level_graph (bool): Indicates whether each signal is one node or not
            core_location_graph (bool): Indicates whether core location graph is used
        """
        super().__init__()

        # Model layers
        if core_location_graph and (not signal_level_graph):
            self.conv1d = Conv1d(in_channels=num_signal_channels,
                                 out_channels=1,
                                 kernel_size=conv1d_kernel_size,
                                 stride=conv1d_stride)
        else:
            self.conv1d = Conv1d(in_channels=1,
                                 out_channels=1,
                                 kernel_size=conv1d_kernel_size,
                                 stride=conv1d_stride)
                                
        self.conv1d_output_size = floor((input_dim - (conv1d_kernel_size-1) - 1)/conv1d_stride+1)

        if conv_type == 'sage':
            self.conv1 = SAGEConv(self.conv1d_output_size,
                                  hidden_dim,
                                  aggregator_type=aggregator_type,
                                  feat_drop=feat_drop)
            self.conv2 = SAGEConv(hidden_dim, int(hidden_dim/2),
                                  aggregator_type=aggregator_type,
                                  feat_drop=feat_drop)
        elif conv_type == 'gcn':
            self.conv1 = GraphConv(self.conv1d_output_size, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, int(hidden_dim/2))
        elif conv_type == 'sg':
            self.conv1 = SGConv(self.conv1d_output_size, hidden_dim)
            self.conv2 = SGConv(hidden_dim, int(hidden_dim/2))
        elif conv_type == 'gat':
            self.conv1 = GATConv(self.conv1d_output_size,
                                 hidden_dim,
                                 feat_drop=feat_drop,
                                 attn_drop=attn_drop,
                                 num_heads=num_heads)
            self.conv2 = GATConv(hidden_dim*num_heads, int(hidden_dim/2), feat_drop=feat_drop, attn_drop=attn_drop,
                                 num_heads=num_heads)

        self.conv_dropout = Dropout2d(p=conv_dropout_p)

        self.fc_1 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc_dropout = nn.Dropout(p=fc_dropout_p)
        self.fc_2 = nn.Linear(int(hidden_dim/4), int(hidden_dim/8))
        self.fc_3 = nn.Linear(int(hidden_dim/8), 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda
        self.apply_output_activation = apply_output_activation
        self.num_signal_channels = num_signal_channels
        self.conv_type = conv_type
        self.signal_level_graph = signal_level_graph

    def forward(self, g):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['h']

        if self.use_cuda:
            h = h.cuda()

        # 1D conv
        if (self.num_signal_channels == 1) or self.signal_level_graph:
            h = torch.unsqueeze(h, dim=1)
        h = F.relu(self.conv1d(h))
        h = torch.squeeze(h)

        # Two layers of Graph Convolution
        h = F.relu(self.conv_dropout(self.conv1(g, h)))

        if self.conv_type == 'gat':
            h = torch.flatten(h, start_dim=1)

        h = F.relu(self.conv2(g, h))

        if self.conv_type == 'gat':
            h = torch.mean(h, dim=1)

        # Fully connected output layer
        h = F.relu(self.fc_dropout(self.fc_1(h)))
        h = F.relu(self.fc_dropout(self.fc_2(h)))
        h = self.fc_3(h)

        if self.apply_output_activation:
            h = self.out_act(h)

        return h


class GraphBinaryClassifier(nn.Module):
    """
    Graph classification model for the prostate cancer dataset
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 aggregator_type='mean',
                 feat_drop=0,
                 use_cuda=False,
                 fc_dropout_p=0,
                 conv_dropout_p=0,
                 attn_drop=0,
                 conv_type='sage',
                 conv1d_kernel_size=10,
                 conv1d_stride=8,
                 num_heads=1,
                 apply_output_activation=False):
        """
        Constructor for the GraphBinaryClassifier class
        Parameters:
            input_dim (int): dimension of features for each node
            hidden_dim (int): dimension of hidden embeddings
            aggregator_type (str): One of mean, lstm, gcn or pool
            feat_drop (float): Indicates the dropout rate for the features
            use_cuda (bool): Indicates whether GPU should be utilized or not
            fc_dropout_p (float): Indicates the FC layer dropout ratio
            attn_drop (float): Indicates the dropout rate for the attention mechanism
            conv_type (str): Type of GNN to use
            conv1d_kernel_size (int): Kernel size for 1D conv
            conv1d_stride (int): Stride used for 1D conv
            num_heads (int): Number of attention heads for the GAT network
            apply_output_activation (bool): Indicates whether sigmoid is applied at the output or not
        """
        super().__init__()

        # Model layers
        self.conv1d = Conv1d(in_channels=1,
                             out_channels=1,
                             kernel_size=conv1d_kernel_size,
                             stride=conv1d_stride)

        self.conv1d_output_size = floor((input_dim - (conv1d_kernel_size-1) - 1)/conv1d_stride+1)

        if conv_type == 'sage':
            self.conv1 = SAGEConv(self.conv1d_output_size,
                                  hidden_dim,
                                  aggregator_type=aggregator_type,
                                  feat_drop=feat_drop)
            self.conv2 = SAGEConv(hidden_dim,
                                  int(hidden_dim/2),
                                  aggregator_type=aggregator_type,
                                  feat_drop=feat_drop)
        elif conv_type == 'gcn':
            self.conv1 = GraphConv(self.conv1d_output_size, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, int(hidden_dim/2))
        elif conv_type == 'sg':
            self.conv1 = SGConv(self.conv1d_output_size, hidden_dim)
            self.conv2 = SGConv(hidden_dim, int(hidden_dim/2))
        elif conv_type == 'gat':
            self.conv1 = GATConv(self.conv1d_output_size,
                                 hidden_dim,
                                 feat_drop=feat_drop,
                                 attn_drop=attn_drop,
                                 num_heads=num_heads)
            self.conv2 = GATConv(hidden_dim*num_heads,
                                 int(hidden_dim/2),
                                 feat_drop=feat_drop,
                                 attn_drop=attn_drop,
                                 num_heads=num_heads)

        self.conv_dropout = Dropout2d(p=conv_dropout_p)

        self.fc_1 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc_dropout = nn.Dropout(p=fc_dropout_p)
        self.fc_2 = nn.Linear(int(hidden_dim/4), int(hidden_dim/8))
        self.fc_3 = nn.Linear(int(hidden_dim/8), 1)
        self.out_act = nn.Sigmoid()

        # Pooling layer
        self.global_pool = AvgPooling()

        self.use_cuda = use_cuda
        self.apply_output_activation = apply_output_activation
        self.conv_type = conv_type

    def forward(self, g, itr=None, label=None, cg=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['h']

        if self.use_cuda:
            h = h.cuda()

        # 1D conv
        h = torch.unsqueeze(h, dim=1)
        h = F.relu(self.conv1d(h))
        h = torch.squeeze(h)

        # Two layers of Graph Convolution
        h = F.relu(self.conv_dropout(self.conv1(g, h)))

        if self.conv_type == 'gat':
            h = torch.flatten(h, start_dim=1)

        h = F.relu(self.conv2(g, h))

        if self.conv_type == 'gat':
            h = torch.mean(h, dim=1)

        # Use the mean of hidden embeddings to find graph embedding
        hg = self.global_pool(g, h)

        # Fully connected output layer
        h = F.relu(self.fc_dropout(self.fc_1(hg)))
        h = F.relu(self.fc_dropout(self.fc_2(h)))
        h = self.fc_3(h)

        if self.apply_output_activation:
            h = self.out_act(h)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                f.write(str(cg) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return h

