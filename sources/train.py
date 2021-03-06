from data import collate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from data import ProstateCancerDataset
import torch
from model import GraphConvBinaryClassifier
from model import GraphAttConvBinaryClassifier
from model import GraphSageBinaryClassifier
from model import GatedGraphConvBinaryClassifier
from model import SimpleGraphConvBinaryClassifier
from model import ChebConvBinaryClassifier
import argparse
from math import floor
from sklearn.metrics import roc_auc_score
from datetime import datetime
import logging


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
    parser.add_argument('--mat_file_path',
                        type=str,
                        required=True,
                        help='Path to time domain mat file')
    parser.add_argument('--fft_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to frequency domain mat file')
    parser.add_argument('--test_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to test time domain mat file')
    parser.add_argument('--test_fft_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to test time domain mat file')
    parser.add_argument('--val_fft_mat_file_path',
                        type=str,
                        default=None,
                        help='Path to validation time domain mat file (Only provide if not splitting training data for '
                             'validation)')
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
    parser.add_argument('--cuda_knn',
                        type=bool,
                        default=False,
                        help='Indicates whether the graph created using GPU')
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
                        default='../',
                        help='Path to save file with loss and accuracy history')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None,
                        help='Path to pickle file to continue the training from')
    parser.add_argument('--gnn_type',
                        type=str,
                        default='gcn',
                        help='GNN type to use for the classifier {gcn, gat, sage, gated, sg, cheb}')
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
    parser.add_argument('--get_cancer_grade',
                        type=bool,
                        default=False,
                        help='Extract cancer grade information from the dataset.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--val_mat_file_path',
                       type=str,
                       default=None,
                       help='Path to validation time domain mat file (Only provide if not splitting training data for '
                            'validation)')
    group.add_argument('--val_split',
                       type=float,
                       default=0.0,
                       help='Validation set split')
    args = parser.parse_args()

    # Common arguments
    mat_file_path = args.mat_file_path
    fft_mat_file_path = args.fft_mat_file_path
    test_mat_file_path = args.test_mat_file_path
    test_fft_mat_file_path = args.test_fft_mat_file_path
    val_mat_file_path = args.val_mat_file_path
    val_fft_mat_file_path = args.val_fft_mat_file_path
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
    cuda_knn = args.cuda_knn
    embeddings_path = args.embeddings_path
    get_cancer_grade = args.get_cancer_grade

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Load the train dataset
    dataset_train = ProstateCancerDataset(mat_file_path=mat_file_path,
                                          train=True,
                                          k=k,
                                          weighted=weighted,
                                          knn_n_jobs=knn_n_jobs,
                                          fft_mat_file_path=fft_mat_file_path,
                                          threshold=threshold,
                                          perform_pca=perform_pca,
                                          num_pca_components=input_dim,
                                          cuda_knn=cuda_knn,
                                          get_cancer_grade=get_cancer_grade)

    dataset_train_len = len(dataset_train)
    logger.info("Training dataset has {} samples".format(dataset_train_len))

    # Load the validation dataset if it's provided separately
    calculate_validation_accuracy = False
    if val_mat_file_path:
        calculate_validation_accuracy = True
        dataset_val = ProstateCancerDataset(mat_file_path=val_mat_file_path,
                                            train=False,
                                            k=k,
                                            weighted=weighted,
                                            knn_n_jobs=knn_n_jobs,
                                            fft_mat_file_path=val_fft_mat_file_path,
                                            threshold=threshold,
                                            perform_pca=perform_pca,
                                            num_pca_components=input_dim,
                                            test_data_string="data_val",
                                            test_fft_data_string="data_val",
                                            test_data_label_string="label_val",
                                            get_cancer_grade=get_cancer_grade,
                                            cancer_grade_string="GS_val",
                                            cuda_knn=cuda_knn)

        val_set_len = len(dataset_val)
        val_data_loader = DataLoader(dataset_val, shuffle=True, collate_fn=collate)

        logger.info("Validation dataset has {} samples.".format(val_set_len))

    # Load the test dataset if provided
    calculate_test_accuracy = False
    if test_mat_file_path:
        calculate_test_accuracy = True
        dataset_test = ProstateCancerDataset(mat_file_path=test_mat_file_path,
                                             train=False,
                                             k=k,
                                             weighted=weighted,
                                             knn_n_jobs=knn_n_jobs,
                                             fft_mat_file_path=test_fft_mat_file_path,
                                             threshold=threshold,
                                             perform_pca=perform_pca,
                                             num_pca_components=input_dim,
                                             test_data_string="data",
                                             test_fft_data_string="FFT_train",
                                             test_data_label_string="label",
                                             get_cancer_grade=get_cancer_grade,
                                             cancer_grade_string="GS",
                                             cuda_knn=cuda_knn)

        test_set_len = len(dataset_test)
        test_data_loader = DataLoader(dataset_test, shuffle=True, collate_fn=collate)

        logger.info("Test dataset has {} samples.".format(test_set_len))

    # Split training data into validation and train set
    if val_mat_file_path is None:
        val_set_len = floor(val_split * dataset_train_len)
        training_set_len = dataset_train_len - val_set_len
        training_set, validation_set = random_split(dataset_train, [training_set_len, val_set_len])

        # Create validation dataset
        if val_set_len is not 0:
            calculate_validation_accuracy = True
            val_data_loader = DataLoader(validation_set, shuffle=True, collate_fn=collate)

        # Create the data loaders for validation and training
        train_data_loader = DataLoader(training_set, shuffle=True, collate_fn=collate)

        logger.info("Using {} samples for the training set and {} points for the validation set.".format(
            training_set_len,
            val_set_len))

    else:
        # Create the data loaders for validation and training
        train_data_loader = DataLoader(dataset_train, shuffle=True, collate_fn=collate)
        training_set_len = len(dataset_train)

        logger.info("Training dataset has {} samples.".format(training_set_len))

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

        logger.info("Resuming from epoch: {} with loss: {}".format(starting_epoch, loss.item()))

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Loss and accuracy variables
    train_losses = []
    train_accs = []
    loss = 0

    if calculate_validation_accuracy:
        max_val_acc = 0
        val_accs = []
        val_losses = []

    if calculate_test_accuracy:
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

            y_true.append(label.detach().item())

            # Predict labels
            if embeddings_path is not None:
                prediction = model(bg,
                                   epoch,
                                   label,
                                   cg,
                                   embeddings_path+"train")
            else:
                prediction = model(bg)

            y_score.append(prediction.detach().item())

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
        logger.info('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        train_losses.append(epoch_loss)
        acc = roc_auc_score(y_true, y_score)
        logger.info("Training accuracy {:.4f}".format(acc))
        train_accs.append(acc)
        t_end = datetime.now()
        logger.info("it took {} for the training epoch to finish".format(t_end-t_start))

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

        # Find validation loss
        if calculate_validation_accuracy:

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

                    y_true.append(label.detach().item())

                    # Predict labels
                    if embeddings_path is not None:
                        prediction = model(bg,
                                           epoch,
                                           label,
                                           cg,
                                           embeddings_path+"val")
                    else:
                        prediction = model(bg)

                    y_score.append(prediction.detach().item())

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

        if calculate_test_accuracy:

            t_start = datetime.now()

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
                else:
                    prediction = model(bg)

                y_score.append(prediction.detach().item())

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
