from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


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

    def forward(self, g):
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

        return self.out_act(out)


class GraphAttConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim, num_heads=1, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_heads (int): Number of attention heads used
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphAttConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g):
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

        return self.out_act(out)


class GraphSageBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GraphSage
    """
    def __init__(self, in_dim, hidden_dim, aggregator_type='mean', use_cuda=False):
        """
        Constructor for the GraphSageBinaryClassifier class
        Parameters:
            in_dim (int): dimension of features for each node
            hidden_dim (int): dimension of hidden embeddings
            aggregator_type (str): One of mean, lstm, gcn or pool
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphSageBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type=aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type=aggregator_type)
        self.fc = nn.Linear(hidden_dim, 1)
        self.out_act = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g):
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

        return self.out_act(out)
