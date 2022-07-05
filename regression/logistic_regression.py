from typing import Callable

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression

import unittest


def log_loss_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape != y_true.shape:
        raise Exception(f'y_pred shape {y_pred.shape} not equal to y_true shape {y_true.shape}')

    n = y_pred.shape[0]
    return (-1 / n) * ((y_true * np.log(y_pred)) +
                       ((1 - y_true) * np.log(1 - y_pred))
                       ).sum()


def logistic_model_log_loss_gradient(
        m_x: np.ndarray,
        w: np.ndarray,
        y_true: np.ndarray) -> np.ndarray:
    """
    Calculates gradient vector for logistic model with form
        y_hat = sigmoid(M_x * w) where M_x is size (i, 1 + p) where (1 + p) the number of properties
        added 1 b/c M_x should be augmented with ones vector horizontally b/c
        that column will represent w_0 (the basis or intercept)

    The gradient follows the same form as the gradient of linear regression with
        cost function as MSE, choosing a loss function as Log Loss is not a coincidence

    :param m_x: ones column with input data (i, 1 + p)
    :param w: parameter vector with size (1 + p)
    :param y_true: target data with size (i, 1)
    :return: a gradient vector which represents the change in LL (log loss) for a change in each weight
    """
    linear_combination = np.matmul(m_x, w)
    return (1 / m_x.shape[0]) * np.matmul(m_x.T, sigmoid(linear_combination) - y_true)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, learning_rate=0.5):
        self.loss_history = []
        self.learning_rate = learning_rate
        self.w = None

    def fit(self,
            input_data: np.ndarray,
            target_data: np.ndarray,
            cost_func_gradient: Callable) -> None:
        """

        :param input_data: data with size (i, p)
        :param target_data: data with size (i, 1)
        :param cost_func_gradient: calculates gradient vector, obtained by taking derivative
            of cost function with model
        :return: None
        """

        ones: np.ndarray = np.ones((input_data.shape[0], 1))
        m_x: np.ndarray = np.hstack((ones, input_data))
        self.w: np.ndarray = np.random.random((m_x.shape[1], 1))

        for _ in range(10000):
            gradient = cost_func_gradient(m_x, self.w, target_data)
            self.w -= self.learning_rate * gradient

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        ones: np.ndarray = np.ones((input_data.shape[0], 1))
        m_x: np.ndarray = np.hstack((ones, input_data))
        return sigmoid(np.matmul(m_x, self.w))


class LogisticRegressionTest(unittest.TestCase):

    def test_against_sklearn(self):
        num_train, num_test = 4000, 1000
        x, y = make_classification(num_train + num_test)
        input_train, target_train = x[:num_train], y[:num_train]
        input_test, target_test = x[num_train:], y[num_train:]

        target_train = np.reshape(target_train, (target_train.shape[0], 1))
        target_test = np.reshape(target_test, (target_test.shape[0], 1))

        model = LogisticRegression()
        model.fit(input_train, target_train, logistic_model_log_loss_gradient)

        predictions = np.vectorize(lambda i: 1 if i > .5 else 0)(model.predict(input_test))
        accuracy = sum(predictions == target_test) / len(target_test)

        sklearn_model = SKLearnLogisticRegression()
        sklearn_model.fit(input_train, target_train)
        sklearn_y_pred = sklearn_model.predict(input_test)
        sklearn_accuracy = sum(sklearn_y_pred == target_test.flatten()) / len(target_test)

        print(f'Model accuracy: {accuracy}\nSklearn model accuracy: {sklearn_accuracy}')

        # within 5% of each other
        self.assertTrue(abs(accuracy - sklearn_accuracy) < .05)
