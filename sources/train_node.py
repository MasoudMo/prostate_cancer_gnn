from data import create_graph_for_node_classification
from model import GraphConvBinaryNodeClassifier
import torch.optim as optim
import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn


logger_level = logging.INFO

logger = logging.getLogger('gnn_prostate_cancer')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    g, labels, cg, trainmsk, valmsk, testmsk = create_graph_for_node_classification(mat_file_path='../data/BK_RF_P1_90_MICCAI_33.mat',
                                                                                    fft_mat_file_path='../data/BK_FFT_P1_90_MICCAI_33.mat',
                                                                                    test_mat_file_path='../data/BK_RF_P91_110.mat',
                                                                                    test_fft_mat_file_path='../data/BK_RF_FFT_resmp_2_100_P91_110.mat',
                                                                                    weighted=False,
                                                                                    k=150,
                                                                                    knn_n_jobs=1,
                                                                                    cuda_knn=False,
                                                                                    threshold=None,
                                                                                    perform_pca=True,
                                                                                    num_pca_components=40,
                                                                                    get_cancer_grade=False)

    labels = torch.squeeze(torch.tensor(labels, dtype=torch.long))

    model = GraphConvBinaryNodeClassifier(in_dim=40, hidden_dim=30, num_classes=2, use_cuda=False)

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    f_train_l = open("../data/node_classification_train_losses.txt", "w")
    f_train_a = open("../data/node_classification_train_accs.txt", "w")
    f_val_l = open("../data/node_classification_val_losses.txt", "w")
    f_val_a = open("../data/node_classification_val_accs.txt", "w")
    f_test_l = open("../data/node_classification_test_losses.txt", "w")
    f_test_a = open("../data/node_classification_test_accs.txt", "w")

    max_val_acc = 0
    
    for epoch in range(1000000):

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

        # Calculate Training Accuracy
        with torch.no_grad():

            model.eval()

            y_pred = np.argmax(prediction[trainmsk].detach(), axis=1)
            acc = accuracy_score(labels[trainmsk], y_pred)
            logger.info("Training accuracy {:.4f}".format(acc))
            f_train_l.write(str(loss.item()) + "\n")
            f_train_a.write(str(acc) + "\n")

            prediction = model(g)
            loss = F.cross_entropy(prediction[valmsk].detach(), labels[valmsk])
            y_pred = np.argmax(prediction[valmsk].detach(), axis=1)
            acc = accuracy_score(labels[valmsk], y_pred)
            logger.info("Validation accuracy: {:.4f} loss: {:.4f}".format(acc, loss.item()))
            f_val_l.write(str(loss.item()) + "\n")
            f_val_a.write(str(acc) + "\n")

            if acc > max_val_acc:
                max_val_acc = acc
                # Save model checkpoint if validation accuracy has increased
                logger.info("Validation accuracy increased. Saving model to {}".format('../data/best_model.pt'))
                torch.save(model.state_dict(), '../data/best_model.pt')

            prediction = model(g)
            loss = F.cross_entropy(prediction[testmsk].detach(), labels[testmsk])
            y_pred = np.argmax(prediction[testmsk].detach(), axis=1)
            acc = accuracy_score(labels[testmsk], y_pred)
            logger.info("Test accuracy {:.4f} loss: {:.4f}".format(acc, loss.item()))
            f_test_l.write(str(loss.item()) + "\n")
            f_test_a.write(str(acc) + "\n")

    f_train_l.close()
    f_train_a.close()
    f_val_l.close()
    f_val_a.close()
    f_test_l.close()
    f_test_a.close()


if __name__ == "__main__":
    main()
