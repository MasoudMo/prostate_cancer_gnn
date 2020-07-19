from sources.model import GraphConvBinaryClassifier
from sources.model import GraphAttConvBinaryClassifier
from sources.model import GraphSageBinaryClassifier
from sources.data import ProstateCancerDataset
from torch.utils.data import DataLoader
from sources.data import collate
import torch
from sklearn.metrics import roc_auc_score
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GCN training on Prostate Cancer dataset')
    parser.add_argument('--input_path', type=str, required=True, help='path to time domain mat file')
    parser.add_argument('--input_fft_path', type=str, required=True, help='path to frequency domain mat file')
    parser.add_argument('--input_dim', type=int, default='200', help='input (RF Signal) dimension')
    parser.add_argument('--batch_size', type=int, default='1', help='Numbers of cores in a batch')
    parser.add_argument('--hidden_dim', type=int, default='100', help='hidden layer dimension')
    parser.add_argument('--knn_algorithm', type=str, default='auto', help='Algorithm used for the knn graph creation')
    parser.add_argument('--k', type=int, default=10, help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--n_jobs', type=int, default=1, help='Indicates the number jobs to deploy for graph creation')
    parser.add_argument('--weighted', type=bool, default=False, help='Indicates whether the graph is weighted or not')
    parser.add_argument('--model_path',
                        type=str,
                        default='../model/model.pt',
                        help='path to save the trained model to.')
    parser.add_argument('--gnn_type',
                        type=str,
                        default='gcn',
                        help='GNN type to use for the classifier {gcn, gat, graphsage}')
    parser.add_argument('--aggregator_type',
                        type=str,
                        default='mean',
                        help='Aggregator used if gnn_type is set to graphsage')
    parser.add_argument('--num_heads',
                        type=int,
                        default=1,
                        help='Indicates the number of attention heads if gnn_type is gat')
    parser.add_argument('--fft_graph',
                        type=bool,
                        default=False,
                        help='Create the graph using the provided fft data')
    args = parser.parse_args()

    # Common arguments
    input_path = args.input_path
    input_fft_path = args.input_fft_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    model_path = args.model_path
    weighted = args.weighted
    batch_size = args.batch_size
    gnn_type = args.gnn_type
    fft_graph = args.fft_graph

    # KNN arguments
    knn_algorithm = args.knn_algorithm
    n_jobs = args.n_jobs
    k = args.k

    # Graphsage argument
    aggregator_type = args.aggregator_type

    # Gat-specific arguments
    num_head = args.num_heads

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Initialize model
    model = None
    if gnn_type == 'gcn':
        model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == 'gat':
        model = GraphAttConvBinaryClassifier(in_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             use_cuda=use_cuda,
                                             num_heads=num_head)
    elif gnn_type == 'graphsage':
        model = GraphSageBinaryClassifier(in_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          use_cuda=use_cuda,
                                          aggregator_type=aggregator_type)

    # Load model
    model.load_state_dict(torch.load(model_path))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Enter evaluation mode
    model.eval()

    # Load test dataset
    test_set = ProstateCancerDataset(input_path,
                                     fft_mat_file_path=input_fft_path,
                                     train=False,
                                     k=k,
                                     weighted=weighted,
                                     n_jobs=n_jobs,
                                     knn_algorithm=knn_algorithm,
                                     fft_graph=fft_graph)
    dataset_len = len(test_set)
    print("Test dataset has {} points".format(dataset_len))

    # Create the dataloader
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate)

    with torch.no_grad():
        y_true = []
        y_score = []
        for bg, label in test_data_loader:
            if use_cuda:
                label = label.cuda()

            y_true.append(label.detach().item())

            # Predict labels
            prediction = model(bg)

            y_score.append(prediction.detach().item())

        # Compute and print accuracy
        acc = roc_auc_score(y_true, y_score)
        print("The accuracy is {}".format(acc))


if __name__ == '__main__':
    main()

