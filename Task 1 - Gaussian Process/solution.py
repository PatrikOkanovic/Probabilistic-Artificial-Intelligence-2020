from typing import Any

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

np.random.seed(1)
torch.manual_seed(1)
## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted >= true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted >= THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted <= true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    reward = W4 * np.logical_and(predicted < THRESHOLD, true < THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """Using InducingPointKernel in order to handle large data sets."""

        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        rank = 10
        X = train_x.numpy()
        induced_points = np.linspace(0, X.shape[0] - 1, num=rank, dtype=np.int)

        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[induced_points, :],
                                                likelihood=likelihood)

    def forward(self, x):
        """Takes in some n×d data x and returns a MultivariateNormal with the prior
        mean and covariance evaluated at x. In other words, we return the vector μ(x) and
        the n×n matrix Kxx representing the prior mean and covariance matrix of the GP"""

        mean_x = self.mean_module(x)
        coovariance_x = self.covar_module(x)
        return MultivariateNormal(mean_x, coovariance_x)

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class Model():

    def initialize(self, train_x, train_y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModel(train_x, train_y, self.likelihood)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def predict(self, test_x):
        test_x = torch.from_numpy(test_x).double()

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # Reference: https://arxiv.org/abs/1803.06058
        with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
            with gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
                preds = self.model(test_x)
        y = preds.loc.numpy() + np.sqrt(torch.diag(preds.covariance_matrix).detach().numpy())
        return y

    def fit_model(self, train_x, train_y):
        train_x = torch.from_numpy(train_x).double()
        train_y = torch.from_numpy(train_y).double()
        self.initialize(train_x, train_y)

        # Casts all floating point parameters and buffers to ``double`` datatype
        self.model.double()

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        training_iterations = 500

        for i in range(training_iterations):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Get output from model
            output = self.model(train_x)
            # Calc loss and backprop derivatives
            loss = -self.mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f outputscale: %.3f   ' % (
                i + 1, training_iterations, loss.item(),
                self.model.base_covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item(),
                self.model.base_covar_module.outputscale.item()))

            self.optimizer.step()


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)
    print(prediction)


if __name__ == "__main__":
    main()
