from data import create_graph_for_core_classification
from model import NodeBinaryClassifier
import torch.optim as optim
import torch
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import configparser
import argparse
import visdom


logger_level = logging.INFO

logger = logging.getLogger('gnn_prostate_cancer')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GNN training on Prostate Cancer dataset (node classification)')
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
                        help='Path to save node embeddings to')
    parser.add_argument('--best_model_path',
                        type=str,
                        default='../model/node_model.pt',
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
    train_params = config['node_classification_train_params']

    mat_file_path = train_params['TimeSeriesMatFilePath']
    input_dim = int(train_params['SignalLength'])
    hidden_dim = int(train_params['HiddenDim'])
    learning_rate = float(train_params['LearningRate'])
    epochs = int(train_params['Epochs'])
    k = int(train_params['NumNodeNeighbours'])
    gnn_type = train_params['GNNType']
    perform_pca = train_params.getboolean('PerformPCA')
    visualize = train_params.getboolean('Visualize')
    get_cancer_grade = train_params.getboolean('GetCancerGrade')
    fc_dropout_p = float(train_params['FCDropout'])
    conv_dropout_p = float(train_params['ConvDropout'])
    num_signals = int(train_params['NumSignals'])

    if train_params['Threshold'] == 'None':
        threshold = None
    else:
        threshold = float(train_params['Threshold'])

    # Use visdom for online visualization
    if visualize:
        vis = visdom.Visdom(port=8150)

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    g, labels, trainmsk, valmsk, testmsk, _cgs = create_graph_for_core_classification(mat_file_path=mat_file_path,
                                                                                      weighted=weighted,
                                                                                      k=k,
                                                                                      knn_n_jobs=knn_n_jobs,
                                                                                      cuda_knn=use_cuda,
                                                                                      threshold=threshold,
                                                                                      perform_pca=perform_pca,
                                                                                      num_pca_components=input_dim,
                                                                                      get_cancer_grade=get_cancer_grade)

    model = NodeBinaryClassifier(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 feat_drop=feat_drop,
                                 use_cuda=use_cuda,
                                 attn_drop=attn_drop,
                                 conv_type=gnn_type,
                                 fc_dropout_p=fc_dropout_p,
                                 conv_dropout_p=conv_dropout_p,
                                 num_signal_channels=num_signals)

    # Initialize loss function and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Loss and accuracy variables
    train_losses = []
    train_accs = []
    loss = 0

    max_val_acc = 0
    val_accs = []
    val_losses = []

    test_accs = []
    test_losses = []
    
    for epoch in range(epochs):

        model.train()

        if use_cuda:
            torch.cuda.empty_cache()
            labels = labels.to(device)

        prediction = model(g)

        loss = loss_func(prediction[trainmsk], labels[trainmsk])

        # Zero gradients
        optimizer.zero_grad()

        # Back propagate
        loss.backward()

        # Do one optimization step
        optimizer.step()

        # Accumulate epoch loss
        logger.info('Training epoch {}, loss {:.4f}'.format(epoch, loss.item()))

        model.eval()

        # Calculate Training Accuracy
        with torch.no_grad():

            train_acc = roc_auc_score(labels[trainmsk], prediction[trainmsk])
            logger.info("Training accuracy {:.4f}".format(train_acc))

            prediction = model(g)
            val_loss = loss_func(prediction[valmsk], labels[valmsk]).item()
            val_acc = roc_auc_score(labels[valmsk], prediction[valmsk])
            logger.info("Validation accuracy: {:.4f} loss: {:.4f}".format(val_acc, val_loss))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                # Save model checkpoint if validation accuracy has increased
                logger.info("Validation accuracy increased. Saving model to {}".format(best_model_path))
                torch.save(model.state_dict(), best_model_path)

            prediction = model(g)
            test_loss = loss_func(prediction[testmsk], labels[testmsk]).item()
            test_acc = roc_auc_score(labels[testmsk], prediction[testmsk])
            logger.info("Test accuracy {:.4f} loss: {:.4f}".format(test_acc, test_loss))

            # Visualize the loss and accuracy
            if visualize:
                vis.line(Y=torch.reshape(torch.tensor(loss.item()), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                         update='append', win='train_loss',
                         opts=dict(title="Train Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(train_acc), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                         update='append', win='train_acc',
                         opts=dict(title="Train Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

                vis.line(Y=torch.reshape(torch.tensor(val_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_loss',
                         opts=dict(title="Validation Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(val_acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_acc',
                         opts=dict(title="Validation Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

                vis.line(Y=torch.reshape(torch.tensor(test_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='test_loss',
                         opts=dict(title="Test Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(test_acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='test_acc',
                         opts=dict(title="Test Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))


if __name__ == "__main__":
    main()
