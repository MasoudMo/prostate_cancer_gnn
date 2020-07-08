from sources.data import collate
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sources.data import ProstateCancerDataset
import torch
from sources.model import GraphConvBinaryClassifier
import argparse
from math import floor


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss):

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss}, "./model_parameters.pt")

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
    parser.add_argument('--model_path', type=str, default='../model/', help='path to save the trained model to.')
    parser.add_argument('--k', type=int, default=10, help='Indicates the number of neighbours used in knn algorithm')
    parser.add_argument('--weighted', type=bool, default=False, help='Indicates whether the graph is weighted or not')
    args = parser.parse_args()

    input_path = args.input_path
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    lr = args.learning_rate
    epochs = args.epochs
    model_path = args.model_path
    val_split = args.val_split
    k = args.k
    weighted = args.weighted
    batch_size = args.batch_size

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()

    # Load the dataset
    dataset = ProstateCancerDataset(input_path, train=True, k=k, weighted=weighted)
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
    val_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Initialize model
    model = GraphConvBinaryClassifier(in_dim=input_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)

    # Move model to GPU if available
    if use_cuda:
        model = model.cuda()

    # Initialize loss function and optimizer
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(epochs):
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
        epoch_loss /= (dataset_len - 50)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        # Save model checkpoint
        save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), loss)

        # Find validation loss
        model.eval()
        with torch.no_grad():
            validation_loss = 0
            for bg, label in val_data_loader:
                # Move label and graph to GPU if available
                if use_cuda:
                    bg, label = bg.cuda(), label.cuda()

                # Predict labels
                prediction = model(bg)

                # Compute loss
                loss = loss_func(prediction, label)

                # Accumulate validation loss
                validation_loss += loss.detach().item()

            # Compute and print validation loss
            validation_loss /= 50
            print('Validation loss {:.4f}'.format(validation_loss))

    # Save the trained model
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
