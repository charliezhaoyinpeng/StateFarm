from numpy import log, dot, e
from numpy.random import rand
from numpy import linalg as LA
import matplotlib.pyplot as plt


class MyLogisticRegression:
    """
    In stead of using sklearn packages, I come up with a new logistic regression model from scratch. The goal of this is to give maximum freedom for user to tune a logistic model.
    In order to achieve a better accuracy, a held-out validation dataset is used to tune hyperparameters (such as iter, lr, tol and eps) in a training model.
    """

    def __init__(self, iter, lr, tol, eps):
        """
        class constructor
        :param iter: number of iteration in the optimizer, e.g. gradient descent
        :param lr: learning rate to update weights
        :param tol: tolerance to stop the iteration
        :param eps: trade-off parameter of the regularization for the loss function
        """
        self.iter = iter
        self.lr = lr
        self.tol = tol
        self.eps = eps

    def print_hyp(self):
        print("================================ Model 1 Hyperparameters ================================")
        print("iteration: %s; learning rate: %s, tolerance: %s; trade-off: %s" % (self.iter, self.lr, self.tol, self.eps))

    def sigmoid(self, z):
        """
        sigmoid function
        :param z: regressiion input
        :return: a sigmoid output in (0,1)
        """
        return 1 / (1 + e ** (-z))

    def loss_function(self, X, y, weights):
        """
        regularized CE loss function for binary classification problem.
        Here regularization takes the form of 2-norm-square. Other forms can be considered as well.
        :param X: feature vector, n by d
        :param y: ground truth labels, n by 1
        :param weights: parameters, d by 1
        :return: regularized CE loss
        """
        z = dot(X, weights)
        return (-sum(y * log(self.sigmoid(z)) + (1 - y) * log(1 - self.sigmoid(z))) + (self.eps / 2) * (LA.norm(X) ** 2)) / len(X)

    def fit(self, X, y, X_val, y_val):
        """
        This function used the Gradient Descent optimizer to iteratively minimize loss. Other optimizer (e.g. SDG, Adam, etc.) can be considered as well.
        Note that there are two ways to complete the iteration: (1) iterative loss meets the tolerance; (2) iteration ends.
        :param X: feature vector
        :param y: ground truth labels
        :return: trained weights
        """
        train_losses = []
        val_losses = []
        weights = rand(X.shape[1])  # random initialization
        N = len(X)
        loss_info = {'train_label': 'Training loss', 'val_label': 'Validation loss',
                     'title': 'LR: Training and Validation loss', 'xlabel': 'Iterations', 'ylabel': 'Loss', 'save': r'../output/result1.png'}

        print("================================ Training ================================")
        print("Now start training...")
        for i in range(self.iter):
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= (self.lr * dot(X.T, y_hat - y) + self.eps * LA.norm(X)) / N

            # Saving Progress
            iter_loss = self.loss_function(X, y, weights)  # training loss at iter i
            val_loss = self.loss_function(X_val, y_val, weights)  # validation loss at iter i
            train_losses.append(iter_loss)
            val_losses.append(val_loss)

            # Training indicator
            if (i + 1) % 200 == 0:
                print("Now we are in the (%s/%s) iteration. Current training loss is: %s and validation loss is %s." % (i + 1, self.iter, iter_loss, val_loss))

            # One of the stopping criteria when validation loss is less than the tolerance
            if val_loss < self.tol:
                print("The validation loss (%s) is lower than the tolerance (%s) and the iteration stops at iteration (%s/%s)." % (val_loss, self.tol, i + 1, self.iter))
                self.plot_train_val(train_losses, val_losses, range(i, i + 1), loss_info)
                break
            if i + 1 == self.iter:
                print("All iterations have been completed. The current validation loss is %s > %s." % (val_loss, self.tol))
                self.plot_train_val(train_losses, val_losses, range(1, self.iter + 1), loss_info)
        self.weights = weights

    def plot_train_val(self, trains, vals, iters, info):
        """
        show a simple plot for training and validation losses over iterations
        :param trains: array
        :param vals: array
        :param iters: number of iterations
        :param info: info dict
        :return: plot show
        """
        plt.plot(iters, trains, 'g', label=info['train_label'])
        plt.plot(iters, vals, 'b', label=info['val_label'])
        plt.title(info['title'])
        plt.xlabel(info['xlabel'])
        plt.ylabel(info['ylabel'])
        plt.ylim([0, 2.5])
        plt.legend()
        plt.savefig(info['save'])
        plt.show()

    def predict(self, X):
        """
        given trained weights and testing features, this function is used to predict labels for testing data.
        :param X: testing features
        :return: a vector contains predictive labels for testing samples
        """
        # Predicting with sigmoid function
        z_prob = dot(X, self.weights)
        # Returning binary result
        z = [1 if i > 0.5 else 0 for i in self.sigmoid(z_prob)]
        return z, self.sigmoid(z_prob)
