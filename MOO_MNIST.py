import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
from epiuc.uncertainty.classification import LeNet_MNIST
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from config import DATALOADER_CONFIGS, BACKEND


def load_mnist(
    vali_size=0.1,
    generator=None,
    loading_configs=DATALOADER_CONFIGS,
    ROOT_PATH="../data",
):
    """
    Load MNIST dataset.
    See `load_image_data` for more details on the arguments.
    :return: MNIST trainloader, valloader, testloader
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])]
    )

    trainset = torchvision.datasets.MNIST(
        root=ROOT_PATH, train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, **loading_configs)

    testset = torchvision.datasets.MNIST(
        root=ROOT_PATH, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=loading_configs["batch_size"],
        shuffle=False,
        num_workers=loading_configs["num_workers"],
        pin_memory=loading_configs["pin_memory"],
    )

    return trainloader, testloader


trainloader, testloader = load_mnist()

lenet_model = LeNet_MNIST(drop_prob=0.5, in_channels=1, image_width=28, image_heigth=28)
ensemble_lenet = Ensemble_Classifier(
    base_model=lenet_model, n_models=20, random_state=42
)
ensemble_lenet.load("models/MNIST/Ensemble_Classification/")
ensemble_lenet.compile(backend=BACKEND)
ensemble_lenet.to(
    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
)

iter_test = iter(testloader)
all_test_images, all_test_labels = [], []
for image_batch, label_batch in iter_test:
    all_test_images.append(image_batch)
    all_test_labels.append(label_batch)
all_test_images = torch.cat(all_test_images, dim=0)
all_test_labels = torch.cat(all_test_labels, dim=0)

X = all_test_images.flatten(start_dim=1).numpy()
y = all_test_labels.numpy()


for idx, image in enumerate(all_test_images):
    if image.shape != (1, 28, 28):
        raise ValueError(
            "Image shape is not correct. Expected (1, 28, 28), got {}".format(
                image.shape
            )
        )
    prob = ensemble_lenet.predict(image.view(1, *image.shape))[0].max()
    if prob < 1:
        print("Image with low probability: ", prob.item())
        print("Image shape: ", image.shape)
        break


# visualize the image

plt.imshow(image.squeeze().numpy(), cmap="gray")
plt.show()
print("Label = ", all_test_labels[idx])

point_of_interest = image.flatten(start_dim=1)


def get_costs_mnist(epiuc_ensemble_model, X, point_of_interest, desired_class=8):
    """
    Calculate the costs for each point in X with respect to the point of interest.
    :param X: The data points
    :param point_of_interest: The point of interest
    :return: An array of costs
    """
    probabilities, uncertainties = epiuc_ensemble_model.predict(
        X.reshape(-1, 1, 28, 28), raw_output=True
    )
    tu, au, eu = torch.tensor_split(uncertainties.T, 3, dim=0)
    probabilities = (
        probabilities.mean(dim=1).detach().numpy()[:, desired_class]
    )  # Convert to log2 probabilities
    distances = np.linalg.norm(X - point_of_interest, axis=1)
    return np.vstack(
        [au.detach().numpy(), eu.detach().numpy(), distances, probabilities]
    ).T


#### PYMOO ####


class BaseProblem(Problem):
    def __init__(
        self,
        ensemble_model,
        point_of_interest,
        min_x,
        max_x,
        n_features,
        n_obj,
        cost_func,
    ):
        super().__init__(
            n_var=n_features,
            n_obj=n_obj,
            n_ieq_constr=1,
            xl=np.repeat(math.floor(min_x) - 1, n_features),
            xu=np.repeat(math.floor(max_x) + 1, n_features),
        )
        self.model = ensemble_model
        self.point_of_interest = point_of_interest
        self.cost_func = cost_func

    def _evaluate(self, x, out, *args, desired_class=1, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")


class Model_Fixed_Decision_Deterministic(BaseProblem):
    def __init__(
        self, ensemble_model, point_of_interest, min_x, max_x, n_features, cost_func
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=1,
            cost_func=cost_func,
        )
        self.model = ensemble_model
        self.point_of_interest = point_of_interest

    def _evaluate(self, x, out, *args, desired_class=1, **kwargs):
        costs = self.cost_func(
            self.model, x, self.point_of_interest, desired_class=desired_class
        )
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f3]
        out["G"] = [g1]


class Model_Fixed_Decision_Stochastic(BaseProblem):
    def __init__(
        self, ensemble_model, point_of_interest, min_x, max_x, n_features, cost_func
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
        )
        self.model = ensemble_model
        self.point_of_interest = point_of_interest

    def _evaluate(self, x, out, *args, desired_class=1, **kwargs):
        costs = self.cost_func(
            self.model, x, self.point_of_interest, desired_class=desired_class
        )
        f1 = costs[:, 0]  # AU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f1, f3]
        out["G"] = [g1]


class Model_Stochastic_Decision_Fixed(BaseProblem):
    def __init__(
        self, ensemble_model, point_of_interest, min_x, max_x, n_features, cost_func
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
        )
        self.model = ensemble_model
        self.point_of_interest = point_of_interest

    def _evaluate(self, x, out, *args, desired_class=1, **kwargs):
        costs = self.cost_func(
            self.model, x, self.point_of_interest, desired_class=desired_class
        )
        f2 = costs[:, 1]  # EU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f2, f3]
        out["G"] = [g1]


class Model_Stochastic_Decision_Stochastic(BaseProblem):
    def __init__(
        self, ensemble_model, point_of_interest, min_x, max_x, n_features, cost_func
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
        )
        self.model = ensemble_model
        self.point_of_interest = point_of_interest

    def _evaluate(self, x, out, *args, desired_class=1, **kwargs):
        costs = self.cost_func(
            self.model, x, self.point_of_interest, desired_class=desired_class
        )
        f1 = costs[:, 0] + costs[:, 1]  # TU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f1, f3]
        out["G"] = [g1]


algorithm = NSGA2(
    pop_size=100,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)


termination = get_termination("n_gen", 10)
##############


########### DEFINITION MINIMIZATION GOALS

m_f_d_d = Model_Fixed_Decision_Deterministic(
    ensemble_lenet,
    point_of_interest.numpy(),
    X.min(),
    X.max(),
    X.shape[-1],
    get_costs_mnist,
)
m_f_d_s = Model_Fixed_Decision_Stochastic(
    ensemble_lenet,
    point_of_interest.numpy(),
    X.min(),
    X.max(),
    X.shape[-1],
    get_costs_mnist,
)
m_s_d_f = Model_Stochastic_Decision_Fixed(
    ensemble_lenet,
    point_of_interest.numpy(),
    X.min(),
    X.max(),
    X.shape[-1],
    get_costs_mnist,
)
m_s_d_s = Model_Stochastic_Decision_Stochastic(
    ensemble_lenet,
    point_of_interest.numpy(),
    X.min(),
    X.max(),
    X.shape[-1],
    get_costs_mnist,
)


#### MINIMIZATION

res_m_f_d_d = minimize(
    m_f_d_d, algorithm, termination=termination, seed=1, save_history=True, verbose=True
)
res_m_f_d_s = minimize(
    m_f_d_s, algorithm, termination=termination, seed=1, save_history=True, verbose=True
)
res_m_s_d_f = minimize(
    m_s_d_f, algorithm, termination=termination, seed=1, save_history=True, verbose=True
)
res_m_s_d_s = minimize(
    m_s_d_s, algorithm, termination=termination, seed=1, save_history=True, verbose=True
)

pareto_points_m_f_d_d = res_m_f_d_d.X
objetives_m_f_d_d = res_m_f_d_d.F

pareto_points_m_f_d_s = res_m_f_d_s.X
objetives_m_f_d_s = res_m_f_d_s.F

pareto_points_m_s_d_f = res_m_s_d_f.X
objetives_m_s_d_f = res_m_s_d_f.F

pareto_points_m_s_d_s = res_m_s_d_s.X
objetives_m_s_d_s = res_m_s_d_s.F


######### VISUALIZE M_F_D_D
plt.imshow(pareto_points_m_f_d_d.reshape(28, 28), cmap="gray")
plt.show()

######### VISUALIZE M_F_D_S
fig, ax = plt.subplots(figsize=(8, 8), ncols=pareto_points_m_f_d_s.shape[0], nrows=1)
for i in range(pareto_points_m_f_d_s.shape[0]):
    ax[i].imshow(pareto_points_m_f_d_s[i].reshape(28, 28), cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(f"Pareto Point {i + 1}")
plt.tight_layout()
plt.show()

######### VISUALIZE M_S_D_D
fig, ax = plt.subplots(figsize=(8, 8), ncols=pareto_points_m_s_d_f.shape[0], nrows=1)
for i in range(pareto_points_m_s_d_f.shape[0]):
    ax[i].imshow(pareto_points_m_s_d_f[i].reshape(28, 28), cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(f"Pareto Point {i + 1}")
plt.tight_layout()
plt.show()


########## VISUALIZE M_S_D_S
fig, ax = plt.subplots(figsize=(8, 8), ncols=pareto_points_m_s_d_s.shape[0], nrows=1)
for i in range(pareto_points_m_s_d_s.shape[0]):
    ax[i].imshow(pareto_points_m_s_d_s[i].reshape(28, 28), cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(f"Pareto Point {i + 1}")
plt.tight_layout()
plt.show()
