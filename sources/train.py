from sources.data import collate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sources.data import ProstateCancerDataset
import torch
from sources.model import GraphConvBinaryClassifier
from sources.model import GraphAttConvBinaryClassifier
from sources.model import GraphSageBinaryClassifier
import argparse
from math import floor
from sklearn.metrics import roc_auc_score


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss, path):
    """
    Saves model checkpoint for later training
    Parameters:
        epoch (int): Current epoch
        model_state_dict (dict): State dictionary obtained from model.state_dict()
        optimizer_state_dict (dict): State dictionary obtained from optimizer.state_dict()
        loss (Tensor): current loss
        path (str): Path to save the checkpoint to
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss}, path)

    return


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GCN training on Prostate Cancer dataset')
    parser.add_argument('--input_path', type=str, required=True, help='path to mat file')
    parser.add_argument('--input_dim', type=int, default='200', help='input (RF Signal) dimension')
    parser.add_argument('--batch_size', type=int, default='1', help='Numbers of cores in a batch')
    parser.add_argument('--hidden_dim', type=int, default='100', help='hidden layer dimension')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='learning rate')
    parser.add_argument('--val_split', type=float, default='0.2', help='validation set split')
    parser.add_argument('--epochs', type=int, default='20', help='number of epochs')
    parser.add_argument('--model_path', type=str, default='../model/model.pt',
                        help='path to save the trained model to.')
    parser.add_argument('--model_param_path', type=str, default='../model/model_parameters.pt',
                        help='path to save the interim model parameters to.')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        help='GNN type to use for the classifier {gcn, gat, graphsage}')
    parser.add_argument('--knn_algorithm', type=str, default='auto', help='Algorithm used for the knn graph creation')
    parser.add_argument('--k', type=int, default=10, help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--n_jobs', type=int, default=1, help='Indicates the number jobs to deploy for graph creation')
    parser.add_argument('--weighted', type=bool, default=False, help='Indicates whether the graph is weighted or not')
    parser.add_argument('--load_checkpoint', type=bool, default=False,
                        help='True if model should resume from checkpoint')
    parser.add_argument('--feat_drop', type=float, default='0',
                        help='Feature dropout rate used if gnn_type is set to graphsage or gat')
    parser.add_argument('--attn_drop', type=float, default='0',
                        help='Attention dropout rate used if gnn_type is set to gat')
    parser.add_argument('--aggregator_type', type=str, default='mean',
                        help='Aggregator used if gnn_type is set to graphsage')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Indicates the number of attention heads if gnn_type is gat')
    args = parser.parse_args()

    # Common arguments
    input_path = args.input_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    lr = args.learning_rate
    epochs = args.epochs
    model_path = args.model_path
    model_param_path = args.model_param_path
    val_split = args.val_split
    weighted = args.weighted
    batch_size = args.batch_size
    load_checkpoint = args.load_checkpoint
    gnn_type = args.gnn_type

    # KNN arguments
    knn_algorithm = args.knn_algorithm
    n_jobs = args.n_jobs
    k = args.k

    # Gat/Graphsage-specific argument
    feat_drop = args.feat_drop

    # Graphsage argument
    aggregator_type = args.aggregator_type

    # Gat-specific arguments
    attn_drop = args.attn_drop
    num_head = args.num_heads

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()

    # Load the dataset
    dataset = ProstateCancerDataset(input_path, train=True, k=k, weighted=weighted, n_jobs=n_jobs,
                                    knn_algorithm=knn_algorithm)
    dataset_len = len(dataset)
    print("dataset has {} data points".format(dataset_len))

    # Split training data into validation and train set
    validation_set_len = floor(val_split * dataset_len)
    training_set_len = dataset_len - validation_set_len
    training_set, validation_set = random_split(dataset, [training_set_len, validation_set_len])
    print("Using {} points as the training set and {} points as the validation set".format(training_set_len,
                                                                                           validation_set_len))

    # Create the data loaders for validation and training
    train_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_data_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Initialize model
    model = None
    if gnn_type == 'gcn':
        model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == 'gat':
        model = GraphAttConvBinaryClassifier(in_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             use_cuda=use_cuda,
                                             num_heads=num_head,
                                             feat_drop=feat_drop,
                                             attn_drop=attn_drop)
    elif gnn_type == 'graphsage':
        model = GraphSageBinaryClassifier(in_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          use_cuda=use_cuda,
                                          aggregator_type=aggregator_type,
                                          feat_drop=feat_drop)

    # Initialize loss function and optimizer
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize starting epoch index
    starting_epoch = 0

    # Load saved parameters if needed
    if load_checkpoint:
        checkpoint = torch.load(model_param_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

        print("Resuming from these values --> epoch:{}  loss: {}".format(starting_epoch, loss.item()))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    for epoch in range(starting_epoch, epochs):

        # Put model in train model
        model.train()

        epoch_loss = 0
        for bg, label in train_data_loader:
            # Move label and graph to GPU if available
            if use_cuda:
                label = label.cuda()

            # Predict labels
            prediction = model(bg)

            # Compute loss
            loss = loss_func(prediction[0], label)

            # Zero gradients
            optimizer.zero_grad()

            # Back propagate
            loss.backward()

            # Do one optimization step
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += loss.detach().item()

        # Find and print average epoch loss
        epoch_loss /= training_set_len
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))

        # Save model checkpoint
        save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, model_param_path)

        # Find validation loss
        model.eval()
        with torch.no_grad():
            y_true = []
            y_score = []
            validation_loss = 0
            for bg, label in val_data_loader:
                # Move label and graph to GPU if available
                if use_cuda:
                    label = label.cuda()

                y_true.append(label)

                # Predict labels
                prediction = model(bg)

                y_score.append(prediction)

                # Compute loss
                loss = loss_func(prediction[0], label)

                # Accumulate validation loss
                validation_loss += loss.detach().item()

            # Compute and print validation loss
            validation_loss /= validation_set_len
            print('Validation loss {:.4f}'.format(validation_loss))
            acc = roc_auc_score(y_true, y_score)
            print("Validation accuracy {:.4f}".format(acc))

    # Save the trained model
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
