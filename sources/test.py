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
    parser = argparse.ArgumentParser(description='GNN training on Prostate Cancer dataset')
    parser.add_argument('--mat_file_path',
                        type=str,
                        required=True,
                        help='Path to time domain mat file')
    parser.add_argument('--fft_mat_file_path',
                        type=str,
                        default=False,
                        help='Path to frequency domain mat file')
    parser.add_argument('--input_dim',
                        type=int,
                        default='200',
                        help='Input signal dimension')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default='100',
                        help='Hidden layer dimension')
    parser.add_argument('--k',
                        type=int,
                        default=10,
                        help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--knn_n_jobs',
                        type=int,
                        default=1,
                        help='Indicates the number jobs to deploy for graph creation')
    parser.add_argument('--weighted',
                        type=bool,
                        default=False,
                        help='Indicates whether the graph is weighted or not')
    parser.add_argument('--model_path',
                        type=str,
                        default='../model/model.pt',
                        help='Path to load the trained model from.')
    parser.add_argument('--gnn_type',
                        type=str,
                        default='gcn',
                        help='GNN type to use for the classifier {gcn, gat, graphsage}')
    parser.add_argument('--threshold',
                        type=float,
                        default=None,
                        help='Threshold value used in graph creation')
    parser.add_argument('--perform_pca',
                        type=bool,
                        default=False,
                        help='Perform PCA reduction (input_dim is used as number of components)')
    args = parser.parse_args()

    # Common arguments
    mat_file_path = args.mat_file_path
    fft_mat_file_path = args.fft_mat_file_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    model_path = args.model_path
    weighted = args.weighted
    gnn_type = args.gnn_type
    perform_pca = args.perform_pca
    knn_n_jobs = args.knn_n_jobs
    k = args.k
    threshold = args.threshold

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Initialize model
    model = None
    if gnn_type == 'gcn':
        model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == 'gat':
        model = GraphAttConvBinaryClassifier(in_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             use_cuda=use_cuda)
    elif gnn_type == 'graphsage':
        model = GraphSageBinaryClassifier(in_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          use_cuda=use_cuda)

    # Load the saved model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Enter evaluation mode
    model.eval()

    # Load test dataset
    test_set = ProstateCancerDataset(mat_file_path,
                                     fft_mat_file_path=fft_mat_file_path,
                                     train=False,
                                     k=k,
                                     weighted=weighted,
                                     knn_n_jobs=knn_n_jobs,
                                     threshold=threshold,
                                     perform_pca=perform_pca,
                                     num_pca_components=input_dim)

    dataset_len = len(test_set)
    print("Test dataset has {} points".format(dataset_len))

    # Create the dataloader
    test_data_loader = DataLoader(test_set, shuffle=True, collate_fn=collate)

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
        print("Test accuracy is {}".format(acc))


if __name__ == '__main__':
    main()

