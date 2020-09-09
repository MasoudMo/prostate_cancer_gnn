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
from datetime import datetime


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
    parser = argparse.ArgumentParser(description='GNN training on Prostate Cancer dataset')
    parser.add_argument('--mat_file_path',
                        type=str,
                        required=True,
                        help='Path to time domain mat file')
    parser.add_argument('--test_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to test time domain mat file')
    parser.add_argument('--test_fft_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to test time domain mat file')
    parser.add_argument('--fft_mat_file_path',
                        type=str,
                        required=True,
                        help='Path to frequency domain mat file')
    parser.add_argument('--input_dim',
                        type=int,
                        default='200',
                        help='Input signal dimension')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default='100',
                        help='Hidden layer dimension')
    parser.add_argument('--learning_rate',
                        type=float,
                        default='0.001',
                        help='Learning rate')
    parser.add_argument('--val_split',
                        type=float,
                        default='0.2',
                        help='Validation set split')
    parser.add_argument('--epochs',
                        type=int,
                        default='20',
                        help='Number of epochs')
    parser.add_argument('--k',
                        type=int,
                        default=10,
                        help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--weighted',
                        type=bool,
                        default=False,
                        help='Indicates whether the graph is weighted or not')
    parser.add_argument('--knn_n_jobs',
                        type=int,
                        default=1,
                        help='Indicates the number jobs to deploy for graph creation')
    parser.add_argument('--best_model_path',
                        type=str,
                        default='../model/model.pt',
                        help='Path to save the best trained model to.')
    parser.add_argument('--history_path',
                        type=str,
                        default='../',
                        help='Path to save file with loss and accuracy history')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None,
                        help='Path to pickle file to continue the training from')
    parser.add_argument('--gnn_type',
                        type=str,
                        default='gcn',
                        help='GNN type to use for the classifier {gcn, gat, graphsage}')
    parser.add_argument('--feat_drop',
                        type=float,
                        default='0',
                        help='Feature dropout rate used if gnn_type is set to graphsage or gat')
    parser.add_argument('--attn_drop',
                        type=float,
                        default='0',
                        help='Attention dropout rate used if gnn_type is set to gat')
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
    test_mat_file_path = args.test_mat_file_path
    test_fft_mat_file_path = args.test_fft_mat_file_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    lr = args.learning_rate
    epochs = args.epochs
    best_model_path = args.best_model_path
    history_path = args.history_path
    checkpoint_path = args.checkpoint_path
    val_split = args.val_split
    weighted = args.weighted
    gnn_type = args.gnn_type
    knn_n_jobs = args.knn_n_jobs
    k = args.k
    threshold = args.threshold
    perform_pca = args.perform_pca
    feat_drop = args.feat_drop
    attn_drop = args.attn_drop

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()

    # Load the train dataset
    dataset_train = ProstateCancerDataset(mat_file_path=mat_file_path,
                                          train=True,
                                          k=k,
                                          weighted=weighted,
                                          knn_n_jobs=knn_n_jobs,
                                          fft_mat_file_path=fft_mat_file_path,
                                          threshold=threshold,
                                          perform_pca=perform_pca,
                                          num_pca_components=input_dim)

    dataset_train_len = len(dataset_train)
    print("training dataset has {} samples".format(dataset_train_len))

    # Load the test dataset if provided
    calculate_test_accuracy = False
    if mat_file_path:
        calculate_test_accuracy = True
        dataset_test = ProstateCancerDataset(mat_file_path=test_mat_file_path,
                                             train=False,
                                             k=k,
                                             weighted=weighted,
                                             knn_n_jobs=knn_n_jobs,
                                             fft_mat_file_path=test_fft_mat_file_path,
                                             threshold=threshold,
                                             perform_pca=perform_pca,
                                             num_pca_components=input_dim)

        test_set_len = len(dataset_test)
        test_data_loader = DataLoader(dataset_test, shuffle=True, collate_fn=collate)

        print("test dataset has {} samples".format(test_set_len))

    # Split training data into validation and train set
    validation_set_len = floor(val_split * dataset_train_len)
    training_set_len = dataset_train_len - validation_set_len
    training_set, validation_set = random_split(dataset_train, [training_set_len, validation_set_len])
    print("Using {} samples for the training set and {} points for the validation set".format(training_set_len,
                                                                                              validation_set_len))

    # Create the data loaders for validation and training
    train_data_loader = DataLoader(training_set, shuffle=True, collate_fn=collate)

    if validation_set_len is not 0:
        val_data_loader = DataLoader(validation_set, shuffle=True, collate_fn=collate)

    # Initialize model
    model = None
    if gnn_type == 'gcn':
        model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == 'gat':
        model = GraphAttConvBinaryClassifier(in_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             use_cuda=use_cuda,
                                             feat_drop=feat_drop,
                                             attn_drop=attn_drop)
    elif gnn_type == 'graphsage':
        model = GraphSageBinaryClassifier(in_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          use_cuda=use_cuda,
                                          feat_drop=feat_drop)

    # Initialize loss function and optimizer
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize starting epoch index
    starting_epoch = 0

    # Load saved parameters if available
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

        print("Resuming from epoch: {} with loss: {}".format(starting_epoch, loss.item()))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Loss and accuracy variables
    max_val_acc = 0
    val_losses = []
    train_losses = []
    val_accs = []
    loss = 0

    if calculate_test_accuracy:
        test_accs = []
        max_test_acc = 0
        test_losses = []

    for epoch in range(starting_epoch, epochs):

        t_start = datetime.now()

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
            loss = loss_func(prediction, label)

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
        print('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        train_losses.append(epoch_loss)

        t_end = datetime.now()
        print("it took {} for the training epoch to finish".format(t_end-t_start))

        # Save to history if needed
        if history_path:
            f = open(history_path+"train_losses.txt", "w")
            for ele in train_losses:
                f.write(str(ele) + "\n")
            f.close()

        # Put model in evaluation mode
        model.eval()

        # Find validation loss
        if validation_set_len is not 0:

            with torch.no_grad():
                y_true = []
                y_score = []
                validation_loss = 0
                # noinspection PyUnboundLocalVariable
                for bg, label in val_data_loader:
                    # Move label and graph to GPU if available
                    if use_cuda:
                        label = label.cuda()

                    y_true.append(label.detach().item())

                    # Predict labels
                    prediction = model(bg)

                    y_score.append(prediction.detach().item())

                    # Compute loss
                    loss = loss_func(prediction, label)

                    # Accumulate validation loss
                    validation_loss += loss.detach().item()

                # Compute and print validation loss
                validation_loss /= validation_set_len
                print('Validation loss {:.4f}'.format(validation_loss))
                val_losses.append(validation_loss)
                acc = roc_auc_score(y_true, y_score)
                print("Validation accuracy {:.4f}".format(acc))
                val_accs.append(acc)

                # Save to history if needed
                if history_path:
                    f = open(history_path + "val_losses.txt", "w")
                    for ele in val_losses:
                        f.write(str(ele) + "\n")
                    f.close()

                    f = open(history_path + "val_accs.txt", "w")
                    for ele in val_accs:
                        f.write(str(ele) + "\n")
                    f.close()

                if acc > max_val_acc:
                    max_val_acc = acc
                    # Save model checkpoint if validation accuracy has increased
                    print("Validation accuracy increased. Saving model to {}".format(best_model_path))
                    save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, best_model_path)

        if calculate_test_accuracy:
            y_true = []
            y_score = []
            test_loss = 0
            # noinspection PyUnboundLocalVariable
            for bg, label in test_data_loader:
                # Move label and graph to GPU if available
                if use_cuda:
                    label = label.cuda()

                y_true.append(label.detach().item())

                # Predict labels
                prediction = model(bg)

                y_score.append(prediction.detach().item())

                # Compute loss
                loss = loss_func(prediction, label)

                # Accumulate validation loss
                test_loss += loss.detach().item()

                # Compute and print validation loss
                # noinspection PyUnboundLocalVariable
                test_loss /= test_set_len
                print('Test loss {:.4f}'.format(test_loss))
                # noinspection PyUnboundLocalVariable
                test_losses.append(test_loss)
                acc = roc_auc_score(y_true, y_score)
                print("Test accuracy {:.4f}".format(acc))
                # noinspection PyUnboundLocalVariable
                test_accs.append(acc)

                # Save to history if needed
                if history_path:
                    f = open(history_path + "test_losses.txt", "w")
                    for ele in test_losses:
                        f.write(str(ele) + "\n")
                    f.close()

                    f = open(history_path + "test_accs.txt", "w")
                    for ele in test_accs:
                        f.write(str(ele) + "\n")
                    f.close()

                if acc > max_test_acc:
                    max_test_acc = acc
                    # Save model checkpoint if validation accuracy has increased
                    print("Test accuracy increased. Saving model to {}".format("test"+best_model_path))
                    save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, "test"+best_model_path)


if __name__ == "__main__":
    main()
