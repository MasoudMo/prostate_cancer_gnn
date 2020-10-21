from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
from dgl.nn.pytorch import GatedGraphConv
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import dgl
import torch


class GraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GCN
    """
    def __init__(self, in_dim, hidden_dim, use_cuda=False):
        """
        Constructor for the GraphConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)


class GatedGraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GCN
    """
    def __init__(self, in_dim, hidden_dim, use_cuda=False):
        """
        Constructor for the GraphConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GatedGraphConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GatedGraphConv(in_dim, hidden_dim, n_steps=10, n_etypes=1)
        self.conv2 = GatedGraphConv(hidden_dim, hidden_dim, n_steps=10, n_etypes=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)


class SimpleGraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GCN
    """
    def __init__(self, in_dim, hidden_dim, use_cuda=False):
        """
        Constructor for the GraphConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(SimpleGraphConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = SGConv(in_dim, hidden_dim)
        self.conv2 = SGConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")

                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)


class GraphAttConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim, feat_drop=0, attn_drop=0, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            feat_drop (float): Indicates the dropout rate for features
            attn_drop (float): Indicates the dropout rate for the attention mechanism
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphAttConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')[0]

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)


class GraphSageBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GraphSage
    """
    def __init__(self, in_dim, hidden_dim, aggregator_type='mean', feat_drop=0, use_cuda=False):
        """
        Constructor for the GraphSageBinaryClassifier class
        Parameters:
            in_dim (int): dimension of features for each node
            hidden_dim (int): dimension of hidden embeddings
            aggregator_type (str): One of mean, lstm, gcn or pool
            feat_drop (float): Indicates the dropout rate for the features
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphSageBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator_type, feat_drop=feat_drop)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type, feat_drop=feat_drop)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)


class ChebConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GCN
    """
    def __init__(self, in_dim, hidden_dim, use_cuda=False):
        """
        Constructor for the GraphConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(ChebConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = ChebConv(in_dim, hidden_dim, k=5)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, k=5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, itr=None, label=None, embedding_path=None):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Fully connected output layer
        out = self.fc(hg)

        # Save graph embeddings to a text file for each epoch
        with torch.no_grad():
            if embedding_path is not None:
                f = open(embedding_path+"_graph_embeddings_itr_" + str(itr) + ".txt", "a+")
                f.write(str(int(label.cpu().numpy()[0][0])) + " ")
                np.savetxt(f, hg.cpu().detach().numpy())
                f.close()

        return self.out_act(out)
