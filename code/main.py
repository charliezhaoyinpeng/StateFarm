from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import os

from data_cleaning import *
from MyLogisticRegression import *
from MyNN import *


def plot_roc(y_true, y_pre_prob, model_name):
    """
    :param y_val: y 1-D ground truth, numpy array
    :param y_val_pred: predicted y 1-d probability vector, numpy array
    :param model_name: model name, str
    :return: roc curve
    """
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pre_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title(model_name + ': Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(r'../output/' + model_name + '_roc.png')
    plt.show()


def Model1(X_train, y_train, X_val, y_val, X_test, out_folder):
    """
    Model 1: Logistic Regression

    :param X_train: training feature vector
    :param y_train: training labels
    :param X_val: validation feature vector
    :param y_val: validation labels
    :param X_test: testing feature vector
    :param out_folder: saving directory
    :return: save testing prediction in to file
    """
    print("================================================================================================")
    print("Now approaching Model 1 (Logistic Regression)...")
    model1 = MyLogisticRegression(iter, lr1, tol, eps)
    model1.fit(X_train, y_train, X_val, y_val)
    pred_val, prob_val = model1.predict(X_val)

    model1.print_hyp()
    print("The Final Validation Accuracy: %s" % accuracy_score(pred_val, y_val))

    plot_roc(y_val, prob_val, 'LR')

    pred_test, pred_test_prob = model1.predict(X_test)
    result = pd.DataFrame(pred_test_prob)
    result.to_csv(out_folder + r'results1.csv', header=False, index=False)


def Model2(X_train, y_train, X_val, y_val, X_test, out_folder):
    """
    Model 2:  A fully connected Neural Network

    :param X_train: training feature vector
    :param y_train: training labels
    :param X_val: validation feature vector
    :param y_val: validation labels
    :param X_test: testing feature vector
    :param out_folder: saving directory
    :return: save testing prediction in to file
    """
    print("================================================================================================")
    print("Now approaching Model 2 (Fully Connected Neural Network)...")

    # convert all input into tensor format
    X_train = torch.tensor(X_train.values)
    y_train = torch.tensor(y_train.values)
    X_val = torch.tensor(X_val.values)
    y_val = torch.tensor(y_val.values)
    X_test = torch.tensor(X_test.values)

    n_in = X_train.shape[1]
    model2 = MyNN(n_in, h_1, h_2)

    # binary cross-entropy loss
    criterion = nn.BCELoss()
    # I choose Adam as the optimizer
    optimizer = torch.optim.Adam(model2.parameters(), lr=lr2, weight_decay=eps2)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model2.train()
        optimizer.zero_grad()
        y_train_pred = model2(X_train.float()).squeeze(1)
        train_loss = criterion(y_train_pred, y_train.float())
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        y_val_pred = model2(X_val.float()).squeeze(1)
        val_loss = criterion(y_val_pred, y_val.float())
        val_losses.append(val_loss.item())
        if (epoch + 1) % 500 == 0:
            print('Epoch %s/%s, training loss: %s and validation loss is: %s' % (epoch + 1, epochs, train_loss.item(), val_loss.item()))

    plt.plot(range(1, epochs + 1), train_losses, 'g', label='Training loss')
    plt.plot(range(1, epochs + 1), val_losses, 'b', label='Validation loss')
    plt.title('NN: Training and Validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ylim([0, 2.5])
    plt.legend()
    plt.savefig(r'../output/result2.png')
    plt.show()
    print("The Final Validation Accuracy: %s:" % accuracy_score(y_val.numpy(), y_val_pred.detach().numpy().round()))

    plot_roc(y_val.numpy(), y_val_pred.detach().numpy(), 'NN')

    with torch.no_grad():
        pred_test_prob = model2(X_test.float()).squeeze(1).data.numpy()
    result = pd.DataFrame(pred_test_prob)
    result.to_csv(out_folder + r'results2.csv', header=False, index=False)


def main(train_path, test_path, out_folder):
    X_train, X_val, y_train, y_val = data_clean(pd.read_csv(train_path), isTrain=True)
    X_test = data_clean(pd.read_csv(test_path), isTrain=False)

    Model1(X_train, y_train, X_val, y_val, X_test, out_folder)
    Model2(X_train, y_train, X_val, y_val, X_test, out_folder)


if __name__ == "__main__":
    # data paths
    data_folder = r'../dataset/'
    out_folder = r'../output/'
    train_path = data_folder + r'exercise_20_train.csv'
    test_path = data_folder + r'exercise_20_test.csv'

    # model1 hyperparameters
    iter = 1200
    lr1 = 0.1
    tol = 0.0001
    eps = 1e-5

    # model2 hyperparameters
    h_1 = 10
    h_2 = 10
    lr2 = 0.0001
    epochs = 7000
    eps2 = 1e-5

    # creat output folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    main(train_path, test_path, out_folder)
