from data import node_classification_core_location_graph, node_classification_knn_graph, ProstateCancerDataset
from model import NodeBinaryClassifier, GraphBinaryClassifier
from data import collate
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import logging
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime
import configparser
import argparse
from math import floor

try:
    import visdom
except ImportError:
    print('Visdom not installed (optional if visualization is not needed)')
    pass

try:
    import wandb
except ImportError:
    print('Wandb not installed (optional if visualization is not needed)')
    pass


# Initialize logger
logger_level = logging.INFO
logger = logging.getLogger('gnn_prostate_cancer')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


def save_checkpoint(epoch, model, optimizer, loss, val_acc, path):
    """
    Saves model checkpoint for later training
    Parameters:
        epoch (int): Current epoch
        model (Pytorch Model): Pytorch model to save
        optimizer (Pytorch Optimizer): Pytorch optimizer to save
        loss (float): Model's loss at checkpoint
        val_acc (float): Model's validation accuracy at checkpoint
        path (str): Path to save the checkpoint to
    """

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_acc': val_acc}, path)

    return


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GNN training on Prostate Cancer dataset (node classification)')
    parser.add_argument('--training_type',
                        type=str,
                        default=None,
                        choices=['graph', 'node'],
                        help='Indicates whether graph or node classification is performed')
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
    parser.add_argument('--visualization_tool',
                        type=str,
                        default=None,
                        choices=['wandb', 'visdom'],
                        help='Tool used for visualization (visdom or wandb)')
    parser.add_argument('--visdom_port',
                        type=int,
                        default=None,
                        help='The port used by Visdom for plotting the data')
    parser.add_argument('--wandb_user',
                        type=str,
                        default=None,
                        help='Wandb user name if wandb is the visualization tool')
    args = parser.parse_args()

    # Common arguments
    training_type = args.training_type
    best_model_path = args.best_model_path
    history_path = args.history_path
    checkpoint_path = args.checkpoint_path
    visualization_tool = args.visualization_tool
    visdom_port = args.visdom_port
    wandb_user = args.wandb_user

    # Check if required arguments are passed
    if (visualization_tool == 'visdom') and (visdom_port is None):
        parser.error('--visualization_tool visdom requires --visdom_port')

    if (visualization_tool == 'wandb') and (wandb_user is None):
        parser.error('--visualization_tool wandb requires --wandb_user')

    # Parse config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    if training_type == 'node':
        train_params = config['node_classification_train_params']
    elif training_type == 'graph':
        train_params = config['graph_classification_train_params']

    # Shared params between node and graph classification
    mat_file_path = train_params['TimeSeriesMatFilePath']
    input_dim = int(train_params['SignalLength'])
    hidden_dim = int(train_params['HiddenDim'])
    learning_rate = float(train_params['LearningRate'])
    epochs = int(train_params['Epochs'])
    k = int(train_params['NumNodeNeighbours'])
    gnn_type = train_params['GNNType']
    perform_pca = train_params.getboolean('PerformPCA')
    visualize = train_params.getboolean('Visualize')
    fc_dropout_p = float(train_params['FCDropout'])
    conv_dropout_p = float(train_params['ConvDropout'])
    use_core_loc = train_params.getboolean('UseCoreLocationGraph')
    conv1d_kernel_size = int(train_params['1DConvKernelSize'])
    conv1d_stride = int(train_params['1DConvStrideSize'])
    num_heads = int(train_params['NumGATHeads'])
    weight_decay = float(train_params['WeightDecay'])
    feat_drop = float(train_params['GNNFeatDrop'])
    attn_drop = float(train_params['GNNAttnDrop'])
    weighted = train_params.getboolean('WeightedGraph')
    lap_enc_dim = int(train_params['LapPositionalDim'])
    if train_params['Threshold'] == 'None':
        threshold = None
    else:
        threshold = float(train_params['Threshold'])

    # Graph specific params
    if training_type == 'graph':
        batch_size = int(train_params['BatchSize'])
    elif training_type == 'node':
        num_signals = int(train_params['NumSignals'])
        signal_level_graph = train_params.getboolean('SignalLevelGraph')

    # Initialize visualization tool
    if visualize:
        if visualization_tool == 'visdom':
            vis = visdom.Visdom(port=visdom_port)
        elif visualization_tool == 'wandb':
            wandb.login()

            if training_type == 'node':
                wandb.init(entity=wandb_user,
                        project='prostate_cancer_node_classification',
                        config={
                                'learning_rate': learning_rate,
                                'architecture': gnn_type,
                                'dataset': 'BK_RF_P1_140_balance__20210203-175808',
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'num_knn_neighbours': k,
                                'fc_dropout_p': fc_dropout_p,
                                'conv_dropout_p': conv_dropout_p,
                                'num_signals': num_signals,
                                'use_core_loc': use_core_loc,
                                'conv1d_kernel_size': conv1d_kernel_size,
                                'conv1d_stride': conv1d_stride,
                                'num_heads': num_heads,
                                'weight_decay': weight_decay,
                                'feat_drop': feat_drop,
                                'attn_drop': attn_drop,
                                'weighted': weighted,
                                'signal_level_graph': signal_level_graph,
                                'lap_enc_dim': lap_enc_dim})
            elif training_type == 'graph':
                wandb.init(entity=wandb_user,
                        project='prostate_cancer_graph_classification',
                        config={
                                'learning_rate': learning_rate,
                                'architecture': gnn_type,
                                'dataset': 'BK_RF_P1_140_balance__20210203-175808',
                                'input_dim': input_dim,
                                'hidden_dim': hidden_dim,
                                'num_knn_neighbours': k,
                                'fc_dropout_p': fc_dropout_p,
                                'conv_dropout_p': conv_dropout_p,
                                'use_core_loc': use_core_loc,
                                'conv1d_kernel_size': conv1d_kernel_size,
                                'conv1d_stride': conv1d_stride,
                                'num_heads': num_heads,
                                'weight_decay': weight_decay,
                                'feat_drop': feat_drop,
                                'attn_drop': attn_drop,
                                'weighted': weighted,
                                'lap_enc_dim': lap_enc_dim})

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Load the dataset and prepare model based on training type
    if training_type == 'node':
        # training, validation and test nodes are all on the same graph
        if use_core_loc:
            g, labels, mask, _cgs = node_classification_core_location_graph(mat_file_path=mat_file_path,
                                                                            perform_pca=perform_pca,
                                                                            num_pca_components=input_dim,
                                                                            num_signals=num_signals,
                                                                            signal_level_graph=signal_level_graph,
                                                                            lap_enc_dim=lap_enc_dim)
        else:
            g, labels, mask, _cgs = node_classification_knn_graph(mat_file_path=mat_file_path,
                                                                  perform_pca=perform_pca,
                                                                  num_pca_components=input_dim,
                                                                  num_signals=num_signals,
                                                                  signal_level_graph=signal_level_graph,
                                                                  lap_enc_dim=lap_enc_dim)

        # Initialize model
        model = NodeBinaryClassifier(input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     feat_drop=feat_drop,
                                     use_cuda=use_cuda,
                                     attn_drop=attn_drop,
                                     conv_type=gnn_type,
                                     fc_dropout_p=fc_dropout_p,
                                     conv_dropout_p=conv_dropout_p,
                                     num_signal_channels=num_signals,
                                     conv1d_kernel_size=conv1d_kernel_size,
                                     conv1d_stride=conv1d_stride,
                                     num_heads=num_heads,
                                     signal_level_graph=signal_level_graph,
                                     core_location_graph=use_core_loc,
                                     lap_enc_dim=lap_enc_dim)
    elif 'graph':

        # Load the train dataset
        dataset_train = ProstateCancerDataset(mat_file_path=mat_file_path,
                                              mode='train',
                                              weighted=weighted,
                                              k=k,
                                              knn_n_jobs=1,
                                              cuda_knn=use_cuda,
                                              threshold=threshold,
                                              perform_pca=perform_pca,
                                              num_pca_components=input_dim,
                                              lap_enc_dim=lap_enc_dim)

        train_set_len = len(dataset_train)
        train_data_loader = DataLoader(dataset_train,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       collate_fn=collate,
                                       drop_last=True)

        logger.info("Training dataset has {} samples".format(train_set_len))

        # Load the validation dataset
        dataset_val = ProstateCancerDataset(mat_file_path=mat_file_path,
                                            mode='val',
                                            weighted=weighted,
                                            k=k,
                                            knn_n_jobs=1,
                                            cuda_knn=use_cuda,
                                            threshold=threshold,
                                            perform_pca=perform_pca,
                                            num_pca_components=input_dim,
                                            lap_enc_dim=lap_enc_dim)

        val_set_len = len(dataset_val)
        val_data_loader = DataLoader(dataset_val,
                                     shuffle=True,
                                     collate_fn=collate,
                                     batch_size=batch_size,
                                     drop_last=True)

        logger.info("Validation dataset has {} samples.".format(val_set_len))

        # Load the test dataset if provided
        dataset_test = ProstateCancerDataset(mat_file_path=mat_file_path,
                                             mode='test',
                                             weighted=weighted,
                                             k=k,
                                             knn_n_jobs=1,
                                             cuda_knn=use_cuda,
                                             threshold=threshold,
                                             perform_pca=perform_pca,
                                             num_pca_components=input_dim,
                                             lap_enc_dim=lap_enc_dim)

        test_set_len = len(dataset_test)
        test_data_loader = DataLoader(dataset_test,
                                      shuffle=True,
                                      collate_fn=collate,
                                      batch_size=batch_size,
                                      drop_last=True)

        logger.info("Test dataset has {} samples.".format(test_set_len))

        # Dictionary holding the dataloader
        dataloaders = {'Training': train_data_loader,
                       'Validation': val_data_loader,
                       'Test': test_data_loader}

        dataset_lens = {'Training': train_set_len,
                        'Validation': val_set_len,
                        'Test': test_set_len}

        # Initialize model
        model = GraphBinaryClassifier(input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      feat_drop=feat_drop,
                                      use_cuda=use_cuda,
                                      attn_drop=attn_drop,
                                      conv_type=gnn_type,
                                      fc_dropout_p=fc_dropout_p,
                                      conv_dropout_p=conv_dropout_p,
                                      conv1d_kernel_size=conv1d_kernel_size,
                                      conv1d_stride=conv1d_stride,
                                      num_heads=num_heads,
                                      lap_enc_dim=lap_enc_dim)

    # Initialize loss function and optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize starting epoch index
    starting_epoch = 0

    # Load saved parameters if available
    max_val_acc = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        max_val_acc = checkpoint['val_acc']

        logger.info("Resuming from epoch: {} with loss: {}".format(starting_epoch, loss))

    # Move model to available device
    model = model.to(device)

    # Open history files if needed
    if history_path:
        f_train_loss = open(history_path + "train_losses.txt", "a")
        f_train_acc = open(history_path + "train_accs.txt", "a")
        f_val_loss = open(history_path + "val_losses.txt", "a")
        f_val_acc = open(history_path + "val_accs.txt", "a")
        f_test_loss = open(history_path + "test_losses.txt", "a")
        f_test_acc = open(history_path + "test_accs.txt", "a")

        f = {'Training': {'loss': f_train_loss,
                          'acc': f_train_acc},
             'Validation': {'loss': f_val_loss,
                            'acc': f_val_acc},
             'Test': {'loss': f_test_loss,
                      'acc': f_test_acc}}

    # Start the training loop
    for epoch in range(starting_epoch, epochs):

        for phase in ['Training', 'Validation', 'Test']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            if use_cuda:
                torch.cuda.empty_cache()

            with torch.set_grad_enabled(phase == 'Training'):

                if training_type == 'node':

                    t_start = datetime.now()

                    # Run the forward path
                    prediction = model(g)
                    prediction = torch.flatten(prediction)

                    # Compute loss
                    loss = loss_func(prediction[mask[phase]], labels[mask[phase]])

                    if phase == 'Training':
                        # Back propagate
                        loss.backward()

                        # Do one optimization step
                        optimizer.step()

                        # Zero parameter gradients
                        optimizer.zero_grad()

                    with torch.no_grad():
                        # Compute accuracy
                        acc = roc_auc_score(labels[mask[phase]].cpu().detach().numpy(),
                                            prediction[mask[phase]].cpu().detach().numpy())

                        # print epoch stat
                        logger.info('{} epoch {}, loss {:.4f}, accuracy {: .4f}'.format(phase,
                                                                                        epoch,
                                                                                        loss.item(),
                                                                                        acc))

                        # Save model checkpoint if validation accuracy has increased
                        if phase == 'Validation':
                            if acc > max_val_acc:
                                max_val_acc = acc
                                logger.warning("Acc increased ({:.4f}). Saving model to {}".format(max_val_acc,
                                                                                                    best_model_path))
                                save_checkpoint(epoch, model, optimizer, loss.item(), acc, best_model_path)

                        # Visualize the loss and accuracy
                        if visualize:
                            if visualization_tool == 'visdom':
                                vis.line(Y=torch.reshape(torch.tensor(loss.item()), (-1, )),
                                            X=torch.reshape(torch.tensor(epoch), (-1, )),
                                            update='append',
                                            win=phase+'_loss',
                                            opts=dict(title=phase+' Loss Per Epoch',
                                                    xlabel='Epoch',
                                                    ylabel='Loss'))

                                vis.line(Y=torch.reshape(torch.tensor(acc), (-1, )),
                                            X=torch.reshape(torch.tensor(epoch), (-1, )),
                                            update='append',
                                            win=phase+'_acc',
                                            opts=dict(title=phase+"Accuracy Per Epoch",
                                                    xlabel="Epoch",
                                                    ylabel="Accuracy"))

                            if visualization_tool == 'wandb':
                                wandb.log({phase+' BCE Loss': loss.item(),
                                            phase+' AUC ROC Accuracy': acc})

                    t_end = datetime.now()
                    logger.info("it took {} for the epoch to finish".format(t_end-t_start))

                    # Save to history if needed
                    if history_path:
                        f[phase]['loss'].write(str(loss.item())+'\n')
                        f[phase]['acc'].write(str(acc)+'\n')

                elif training_type == 'graph':

                    t_start = datetime.now()

                    y_true = np.empty((0, 3))
                    y_score = np.empty((0, 3))

                    epoch_loss = 0

                    for bg, label, cg in dataloaders[phase]:

                        label = label.to(device)
                        bg = bg.to(device)

                        # Keep track of true label
                        y_true = np.append(y_true, label.cpu().detach().numpy())

                        prediction = model(bg)
                        prediction = torch.flatten(prediction)

                        # Keep track of predicted scores
                        y_score = np.append(y_score, prediction.cpu().detach().numpy())

                        # Compute loss
                        loss = loss_func(prediction, label)

                        if phase == 'Training':
                            # Back propagate
                            loss.backward()
                            
                            # Do one optimization step
                            optimizer.step()

                            # Zero parameter gradients
                            optimizer.zero_grad()

                        # Accumulate epoch loss
                        epoch_loss += loss.detach().item()

                    # Find and print average epoch loss
                    epoch_loss /= floor(dataset_lens[phase]/batch_size)
                    acc = roc_auc_score(y_true, y_score)

                    # Print epoch stat
                    logger.info('{} epoch {}, loss {:.4f}, accuracy {: .4f}'.format(phase, epoch, epoch_loss, acc))

                    # Save model checkpoint if validation accuracy has increased
                    if phase == 'Validation':
                        if acc > max_val_acc:
                            max_val_acc = acc
                            logger.warning("Acc increased ({:.4f}). Saving model to {}".format(max_val_acc,
                                                                                                best_model_path))
                            save_checkpoint(epoch, model, optimizer, loss.item(), acc, best_model_path)

                    # Print elapsed time
                    t_end = datetime.now()
                    logger.info("it took {} for the {} set.".format(t_end - t_start, phase))

                    # Visualize the loss and accuracy
                    if visualize:
                        if visualization_tool == 'visdom':
                            vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1, )),
                                        X=torch.reshape(torch.tensor(epoch), (-1, )),
                                        update='append',
                                        win=phase+'_loss',
                                        opts=dict(title=phase+' Loss Per Epoch',
                                                xlabel='Epoch',
                                                ylabel='Loss'))

                            vis.line(Y=torch.reshape(torch.tensor(acc), (-1, )),
                                        X=torch.reshape(torch.tensor(epoch), (-1, )),
                                        update='append',
                                        win=phase+'_acc',
                                        opts=dict(title=phase+"Accuracy Per Epoch",
                                                xlabel="Epoch",
                                                ylabel="Accuracy"))

                        if visualization_tool == 'wandb':
                            wandb.log({phase+' BCE Loss': epoch_loss,
                                        phase+' AUC ROC Accuracy': acc})

                    # Save to history if needed
                    if history_path:
                        f[phase]['loss'].write(str(loss.item())+'\n')
                        f[phase]['acc'].write(str(acc)+'\n')

    # Close files
    f['Training']['loss'].close()
    f['Training']['acc'].close()
    f['Validation']['loss'].close()
    f['Validation']['acc'].close()
    f['Test']['loss'].close()
    f['Test']['acc'].close()

    # Finish wandb session
    if visualization_tool == 'wandb':
        wandb.finish()


if __name__ == "__main__":
    main()
