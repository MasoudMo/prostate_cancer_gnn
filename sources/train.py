from data import collate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import ProstateCancerDataset
import torch
from model import GraphConvBinaryClassifier
from model import GraphAttConvBinaryClassifier
from model import GraphSageBinaryClassifier
from model import GatedGraphConvBinaryClassifier
from model import SimpleGraphConvBinaryClassifier
from model import ChebConvBinaryClassifier
import argparse
from sklearn.metrics import roc_auc_score
from datetime import datetime
import logging
import visdom
import configparser


logger_level = logging.INFO

logger = logging.getLogger('gnn_prostate_cancer')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


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
    parser.add_argument('--weighted',
                        type=bool,
                        default=False,
                        help='Indicates whether the graph is weighted or not')
    parser.add_argument('--knn_n_jobs',
                        type=int,
                        default=1,
                        help='Indicates the number jobs to deploy for graph creation')
    parser.add_argument('--embeddings_path',
                        type=str,
                        default=None,
                        help='Path to save graph embeddings to')
    parser.add_argument('--best_model_path',
                        type=str,
                        default='../model/model.pt',
                        help='Path to save the best trained model to.')
    parser.add_argument('--history_path',
                        type=str,
                        default=None,
                        help='Path to save file with loss and accuracy history')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None,
                        help='Path to pickle file to continue the training from')
    parser.add_argument('--feat_drop',
                        type=float,
                        default='0',
                        help='Feature dropout rate used if gnn_type is set to graphsage or gat')
    parser.add_argument('--attn_drop',
                        type=float,
                        default='0',
                        help='Attention dropout rate used if gnn_type is set to gat')
    args = parser.parse_args()

    # Common arguments
    best_model_path = args.best_model_path
    history_path = args.history_path
    checkpoint_path = args.checkpoint_path
    weighted = args.weighted
    knn_n_jobs = args.knn_n_jobs
    feat_drop = args.feat_drop
    attn_drop = args.attn_drop
    embeddings_path = args.embeddings_path

    # Parse config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_params = config['train_params']

    mat_file_path = train_params['TimeSeriesMatFilePath']
    input_dim = int(train_params['SignalLength'])
    hidden_dim = int(train_params['HiddenDim'])
    learning_rate = float(train_params['LearningRate'])
    epochs = int(train_params['Epochs'])
    k = int(train_params['NumNodeNeighbours'])
    gnn_type = train_params['GNNType']
    perform_pca = train_params.getboolean('PerformPCA')
    visualize = train_params.getboolean('Visualize')

    if train_params['Threshold'] == 'None':
        threshold = None
    else:
        threshold = float(train_params['Threshold'])

    # Use visdom for online visualization
    if visualize:
        vis = visdom.Visdom()

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # Load the train dataset
    dataset_train = ProstateCancerDataset(mat_file_path=mat_file_path,
                                          mode='train',
                                          weighted=weighted,
                                          k=k,
                                          knn_n_jobs=knn_n_jobs,
                                          cuda_knn=use_cuda,
                                          threshold=threshold,
                                          perform_pca=perform_pca,
                                          num_pca_components=input_dim)

    dataset_train_len = len(dataset_train)
    train_data_loader = DataLoader(dataset_train, shuffle=True, batch_size=3, collate_fn=collate)
    logger.info("Training dataset has {} samples".format(dataset_train_len))

    # Load the validation dataset
    dataset_val = ProstateCancerDataset(mat_file_path=mat_file_path,
                                        mode='val',
                                        weighted=weighted,
                                        k=k,
                                        knn_n_jobs=knn_n_jobs,
                                        cuda_knn=use_cuda,
                                        threshold=threshold,
                                        perform_pca=perform_pca,
                                        num_pca_components=input_dim)

    val_set_len = len(dataset_val)
    val_data_loader = DataLoader(dataset_val, shuffle=True, collate_fn=collate)
    logger.info("Validation dataset has {} samples.".format(val_set_len))

    # Load the test dataset if provided
    dataset_test = ProstateCancerDataset(mat_file_path=mat_file_path,
                                         mode='test',
                                         weighted=weighted,
                                         k=k,
                                         knn_n_jobs=knn_n_jobs,
                                         cuda_knn=use_cuda,
                                         threshold=threshold,
                                         perform_pca=perform_pca,
                                         num_pca_components=input_dim)

    test_set_len = len(dataset_test)
    test_data_loader = DataLoader(dataset_test, shuffle=True, collate_fn=collate)
    logger.info("Test dataset has {} samples.".format(test_set_len))

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
    elif gnn_type == 'sage':
        model = GraphSageBinaryClassifier(in_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          use_cuda=use_cuda,
                                          feat_drop=feat_drop)
    elif gnn_type == "cheb":
        model = ChebConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == "gated":
        model = GatedGraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
    elif gnn_type == "sg":
        model = SimpleGraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)

    # Initialize loss function and optimizer
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize starting epoch index
    starting_epoch = 0

    # Load saved parameters if available
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

        logger.info("Resuming from epoch: {} with loss: {}".format(starting_epoch, loss.item()))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Loss and accuracy variables
    train_losses = []
    train_accs = []
    loss = 0

    max_val_acc = 0
    val_accs = []
    val_losses = []

    test_accs = []
    test_losses = []

    for epoch in range(starting_epoch, epochs):

        t_start = datetime.now()

        y_true = []
        y_score = []

        # Put model in train model
        model.train()

        epoch_loss = 0
        for bg, label, cg in train_data_loader:
            # Move label and graph to GPU if available
            if use_cuda:
                torch.cuda.empty_cache()
                label = label.to(device)

            y_true.append(label.detach())

            # Predict labels
            if embeddings_path is not None:
                prediction = model(bg,
                                   epoch,
                                   label,
                                   cg,
                                   embeddings_path+"train")
                prediction = torch.flatten(prediction)
            else:
                prediction = model(bg)
                prediction = torch.flatten(prediction)

            y_score.append(prediction.detach())

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
        epoch_loss /= dataset_train_len
        logger.info('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        train_losses.append(epoch_loss)
        acc = roc_auc_score(y_true, y_score)
        logger.info("Training accuracy {:.4f}".format(acc))
        train_accs.append(acc)

        t_end = datetime.now()
        logger.info("it took {} for the training epoch to finish".format(t_end-t_start))

        # Visualize the loss and accuracy
        if visualize:
            vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                     update='append', win='tain_loss',
                     opts=dict(title="Train Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

            vis.line(Y=torch.reshape(torch.tensor(acc), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                     update='append', win='train_acc',
                     opts=dict(title="Train Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

        # Save to history if needed
        if history_path:
            f = open(history_path+"train_losses.txt", "w")
            for ele in train_losses:
                f.write(str(ele) + "\n")
            f.close()

            f = open(history_path + "train_accs.txt", "w")
            for ele in train_accs:
                f.write(str(ele) + "\n")
            f.close()

        # Put model in evaluation mode
        model.eval()
        t_start = datetime.now()

        with torch.no_grad():
            y_true = []
            y_score = []
            validation_loss = 0
            # noinspection PyUnboundLocalVariable
            for bg, label, cg in val_data_loader:
                # Move label and graph to GPU if available
                if use_cuda:
                    torch.cuda.empty_cache()
                    label = label.to(device)

                y_true.append(label.detach())

                # Predict labels
                if embeddings_path is not None:
                    prediction = model(bg,
                                       epoch,
                                       label,
                                       cg,
                                       embeddings_path+"val")
                    prediction = torch.flatten(prediction)
                else:
                    prediction = model(bg)
                    prediction = torch.flatten(prediction)

                y_score.append(prediction.detach())

                # Compute loss
                loss = loss_func(prediction, label)

                # Accumulate validation loss
                validation_loss += loss.detach().item()

            # Compute and print validation loss
            validation_loss /= val_set_len
            logger.info('Validation loss {:.4f}'.format(validation_loss))
            # noinspection PyUnboundLocalVariable
            val_losses.append(validation_loss)
            acc = roc_auc_score(y_true, y_score)
            logger.info("Validation accuracy {:.4f}".format(acc))
            # noinspection PyUnboundLocalVariable
            val_accs.append(acc)

            # Print elapsed time
            t_end = datetime.now()
            logger.info("it took {} for the validation set.".format(t_end - t_start))

            # Visualize the loss and accuracy
            if visualize:
                vis.line(Y=torch.reshape(torch.tensor(validation_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_loss',
                         opts=dict(title="Validation Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_acc',
                         opts=dict(title="Validation Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

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
                logger.info("Validation accuracy increased. Saving model to {}".format(best_model_path))
                save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss, best_model_path)

        t_start = datetime.now()

        with torch.no_grad():
            y_true = []
            y_score = []
            test_loss = 0
            # noinspection PyUnboundLocalVariable
            for bg, label, cg in test_data_loader:
                # Move label and graph to GPU if available
                if use_cuda:
                    torch.cuda.empty_cache()
                    label = label.to(device)

                y_true.append(label.detach().item())

                # Predict labels
                if embeddings_path is not None:
                    prediction = model(bg,
                                       epoch,
                                       label,
                                       cg,
                                       embeddings_path+"test")
                    prediction = torch.flatten(prediction)
                else:
                    prediction = model(bg)
                    prediction = torch.flatten(prediction)

                y_score.append(prediction.detach())

                # Compute loss
                loss = loss_func(prediction, label)

                # Accumulate validation loss
                test_loss += loss.detach().item()

            # Compute and print validation loss
            # noinspection PyUnboundLocalVariable
            test_loss /= test_set_len
            logger.info('Test loss {:.4f}'.format(test_loss))
            # noinspection PyUnboundLocalVariable
            test_losses.append(test_loss)
            acc = roc_auc_score(y_true, y_score)
            logger.info("Test accuracy {:.4f}".format(acc))
            # noinspection PyUnboundLocalVariable
            test_accs.append(acc)

            # Print elapsed time
            t_end = datetime.now()
            logger.info("it took {} for the test set.".format(t_end - t_start))

            # Visualize the loss and accuracy
            if visualize:
                vis.line(Y=torch.reshape(torch.tensor(test_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='test_loss',
                         opts=dict(title="Test Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='test_acc',
                         opts=dict(title="Test Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

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


if __name__ == "__main__":
    main()
