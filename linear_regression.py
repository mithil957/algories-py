from typing import Callable

import numpy as np
from numpy.linalg import inv

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinearRegressionSKLearn
from sklearn.metrics import mean_squared_error as mean_squared_error_sklearn

import unittest


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape != y_true.shape:
        raise Exception(f'y_pred shape {y_pred.shape} not equal to y_true shape {y_true.shape}')

    n = max(y_pred.shape)
    return np.square(y_pred - y_true).sum() / n


def linear_model_mean_squared_error_gradient(
        m_x: np.ndarray,
        w: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Calculates gradient vector for linear model with form
        y_hat = M_x * w where M_x is size (i, 1 + p) where p is number of properties
        added 1 b/c M_x should be augmented to have all ones in the first column b/c
        it will be used to calculate w_0
    :param m_x: ones column plus input data with size (i, 1 + p)
    :param w: weights vector with size (p, 1)
    :param y: target data with size (i, 1)
    :return: a gradient vector which represents the change in MSE (mean squared error) for a change in each weight
    """
    return (2 / m_x.shape[0]) * np.matmul(m_x.T, np.matmul(m_x, w) - y)


def linear_model_mean_squared_error_normals(
        m_x: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Gradient equation for MSE with linear model -> d/dw = (2/N) * (M.T) * (M*w - y)
    For a gradient vector for MSE with a linear model, the weights can be calculated exactly by setting
    the gradient equation to zero and solving for w
    Known as normals equation -> w = (M.T * M)^(-1) * M.T * y
    :param m_x: ones column plus input data with size (i, 1 + p)
    :param y: target data with size (i, 1)
    :return: weights for linear model
    """
    matrix_transpose_multiplication = np.matmul(m_x.T, m_x)
    invert_it = inv(matrix_transpose_multiplication)
    invert_transpose_multiplication = np.matmul(invert_it, m_x.T)
    return np.matmul(invert_transpose_multiplication, y)


class LinearRegression:
    """
    Calculates parameters for best fit using gradient descent
    - X: feature data
    - y: target data

    - model/y_pred = X*w
        - it's called linear regression b/c we predict y by multiplying X (our data) by weights w
        - also seen as y_pred = mx + b for one variable but for n variables a matrix is handy
    - cost_func: an estimate of how good our prediction is
        - MSE is common -> 1/n * sum((y_pred - y_true)**2)
    - cost_func_gradient: vector obtained by taking the derivative of cost_func w.r.t parameters
        - d/d(w) of 1/n * sum((X*w - y_true)**2) where X*w is y_pred and w is parameters represented as a vector
    - learning rate: how big a step we want to take in the direction that
        minimizes the cost_func, scales the size of the gradient vector

    Thinking beyond linear regression and looking at any type of regression
    - we need X (input) and y (target)
    - we use a model/equation to transform X to be as "close" to y as possible
    - the model can be any combination of operations on X and parameters
        - model(X, parameters) -> transformed X, which we use as prediction
    - we measure "closeness" with a cost function
        - C(prediction, target) -> number, C(model(X, parameters), target)
    - to make our model better, we update parameters
    - we update parameters based on a gradient vector
    - the gradient vector is calculated by taking the derivative of C
        - d/d(parameters) of C( model(X, parameters), target)

    Quirks:
        finds local minima/maxima
            - many ways to "steer" parameters
        won't work if C is not differentiable
        choice of model and cost function is subjective
            - we chose a way to explain the data
            - we chose a measure of how good our explanation is
    """

    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.w = None

    def fit(self,
            input_data: np.ndarray,
            target_data: np.ndarray,
            cost_func_gradient: Callable) -> None:
        """

        :param input_data: data with size (i, p)
        :param target_data: data with size (i, 1)
        :param cost_func_gradient: calculates gradient vector, obtained by taking derivative of cost function with model
        :return: None
        """
        ones: np.ndarray = np.ones((input_data.shape[0], 1))
        m_x: np.ndarray = np.hstack((ones, input_data))  # size of (i, p + 1)
        self.w: np.ndarray = np.zeros((m_x.shape[1], 1))  # size of (p + 1, 1)

        # 50 is picked arbitrarily
        for _ in range(10000):
            gradient = cost_func_gradient(m_x, self.w, target_data)
            self.w -= self.learning_rate * gradient  # subtract b/c we want to minimize

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        ones: np.ndarray = np.ones((input_data.shape[0], 1))
        m_x: np.ndarray = np.hstack((ones, input_data))
        return np.matmul(m_x, self.w)


class LinearRegressionTest(unittest.TestCase):
    def test_against_sklearn(self):
        """
        Load diabetes dataset
        Train model written above on training data and sklearn model
        Compare loss on test data, if within a threshold, test passes
        """
        data = load_diabetes()
        input_data = data.data
        target_data = data.target
        input_train, input_test, target_train, target_test = train_test_split(
            input_data,
            target_data,
            test_size=0.2,
            random_state=23
        )
        target_train = np.reshape(target_train, (target_train.shape[0], 1))
        target_test = np.reshape(target_test, (target_test.shape[0], 1))

        model = LinearRegression()
        model.fit(input_train,
                  target_train,
                  linear_model_mean_squared_error_gradient)

        y_pred = model.predict(input_test)
        model_loss = mean_squared_error(y_pred, target_test)

        sklearn_model = LinearRegressionSKLearn()
        sklearn_model.fit(input_train, target_train)
        sklearn_y_pred = sklearn_model.predict(input_test)
        sklearn_model_loss = mean_squared_error_sklearn(sklearn_y_pred, target_test)

        print(f'Model loss: {model_loss} \nSKLearn model loss: {sklearn_model_loss}')

        # 50 is picked arbitrarily
        # there are better ways to compare how equivalent two models are
        self.assertTrue(abs(model_loss - sklearn_model_loss) < 50)
