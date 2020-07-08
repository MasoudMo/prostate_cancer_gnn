from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


class GraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset
    """
    def __init__(self, in_dim, hidden_dim, use_cuda=False):
        """
        Constructor for the GraphConvBinaryClassifier class
        Parameters:
            in_dim (int): dimension of features for each node
            hidden_dim (int): dimension of hidden embeddings
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
