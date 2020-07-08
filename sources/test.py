from sources.model import GraphConvBinaryClassifier
from sources.data import ProstateCancerDataset
from torch.utils.data import DataLoader
from sources.data import collate
import torch
from sklearn.metrics import roc_auc_score
import argparse


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Testing the GCN model on Prostate Cancer dataset')
    parser.add_argument('--input_path', type=str, required=True, help='path to mat file')
    parser.add_argument('--input_dim', type=int, default='200', help='input (RF Signal) dimension')
    parser.add_argument('--batch_size', type=int, default='1', help='Numbers of cores in a batch')
    parser.add_argument('--hidden_dim', type=int, default='100', help='hidden layer dimension')
    parser.add_argument('--model_path', type=str, default='../model/', help='path to save the trained model to.')
    parser.add_argument('--k', type=int, default=10, help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--weighted', type=bool, default=False, help='Indicates whether the graph is weighted or not')
    args = parser.parse_args()

    input_path = args.input_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    model_path = args.model_path
    k = args.k
    weighted = args.weighted
    batch_size = args.batch_size

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Initialize the classifier
    model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim)

    # Load model
    model.load_state_dict(torch.load(model_path))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Enter evaluation mode
    model.eval()

    # Load test dataset
    test_set = ProstateCancerDataset(input_path, train=True, k=k, weighted=weighted)

    # Create the dataloader
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate)

    with torch.no_grad():
        y_true = []
        y_score = []
        for bg, label in test_data_loader:
            y_true.append(label)

            # Predict labels
            prediction = model(bg)

            y_score.append(prediction)

        # Compute and print accuracy
        acc = roc_auc_score(y_true, y_score)
        print("The accuracy is {}".format(acc))


if __name__ == '__main__':
    main()

