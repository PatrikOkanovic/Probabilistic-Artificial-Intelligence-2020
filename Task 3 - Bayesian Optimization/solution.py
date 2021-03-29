import numpy as np
import GPy
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
import torch

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        sigma_f = 0.15
        sigma_v = 0.0001
        f_var = 0.5
        f_len = 0.5
        v_mean = 1.5
        v_var = np.sqrt(2)
        v_len = 0.5
        self.v_min = 1.2
        self.xi = 0.01

        x = np.array([[BO_algo.rand_number_in_domain()], [BO_algo.rand_number_in_domain()],
                      [BO_algo.rand_number_in_domain()]])
        f = np.array(np.random.uniform(low=0.0, high=1, size=3)).reshape(3, 1)
        v = np.array([[1.1], [1.05], [1.12]])

        kernel_f = GPy.kern.Matern52(input_dim=domain.shape[0], variance=f_var, lengthscale=f_len)
        kernel_v = GPy.kern.Matern52(input_dim=domain.shape[0], variance=v_var, lengthscale=v_len)

        self.gp_f = GPy.models.GPRegression(x, f, kernel=kernel_f, noise_var=sigma_f)
        self.gp_v = GPy.models.GPRegression(x, v, kernel=kernel_v, noise_var=sigma_v,
                                            mean_function=GPy.mappings.constant.Constant(input_dim=1,
                                                                                         output_dim=1,
                                                                                         value=v_mean))

    @staticmethod
    def rand_number_in_domain():
        array = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
        return array[0]

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximizes the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        x = np.array([x])
        mu_f, var_f = self.gp_f.predict(x)
        mu_f, var_f = mu_f[0], var_f[0]
        xmax, ymax = self.get_best_value()
        sigma_f = np.sqrt(var_f)
        dist = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([1.]))
        Z = (mu_f - ymax - self.xi) / sigma_f
        Z = torch.from_numpy(Z)
        idx = sigma_f == 0
        phi = np.exp(dist.log_prob(Z)) * sigma_f
        phi = phi.numpy()
        ei = (mu_f - ymax - self.xi) * dist.cdf(Z).numpy() + phi
        ei[idx] = 0

        mu_v, sigma_v = self.gp_v.predict(x)
        mu_v, sigma_v = mu_v[0], sigma_v[0]
        constraint = norm(loc=mu_v, scale=sigma_v)
        prob_c = 1.0 - constraint.cdf(self.v_min)  # Pr(C(x)>=1.2)

        if mu_v > self.v_min:
            return ei.mean() * prob_c
        else:
            return prob_c

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        old_x = self.gp_f.X
        old_f = np.array(self.gp_f.Y)
        old_v = np.array(self.gp_v.Y)
        new_x = np.concatenate((old_x, x))
        new_f = np.concatenate((old_f, np.array([f])))
        new_v = np.concatenate((old_v, np.array([v])))
        self.gp_f.set_XY(new_x, new_f)
        self.gp_v.set_XY(new_x, new_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        mask = np.array(self.gp_v.Y) >= self.v_min
        new_y = np.array(self.gp_f.Y)[mask]
        new_x = self.gp_f.X[mask]
        if len(new_y) == 0:
            idx = np.array(self.gp_f.Y).argmax()
            xmax, ymax = self.gp_f.X[idx], self.gp_f.Y[idx]
        else:
            idx = new_y.argmax()
            xmax, ymax = new_x[idx], new_y[idx]
        # if len(self.gp_f.Y) == 1:
        #     xmax, ymax = self.gp_f.X[idx], self.gp_f.Y[idx]
        # else:
        #     xmax, ymax = self.gp_f.X[0][idx], self.gp_f.Y[idx]

        return xmax

    def get_best_value(self):
        idx = np.array(self.gp_f.Y).argmax()
        xmax, ymax = self.gp_f.X[idx], self.gp_f.Y[idx]
        # if len(self.gp_f.Y) == 1:
        #     xmax, ymax = self.gp_f.X[idx], self.gp_f.Y[idx]
        # else:
        #     xmax, ymax = self.gp_f.X[0][idx], self.gp_f.Y[idx]
        
        return xmax, ymax


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
