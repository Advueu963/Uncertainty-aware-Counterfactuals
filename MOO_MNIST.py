import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings as warning

from epiuc.uncertainty.classification import LeNet_MNIST
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.core.mutation import Mutation

from pymoo.optimize import minimize
from config import DATALOADER_CONFIGS, BACKEND
from train_vae_mnist import VAE

parser = argparse.ArgumentParser()
parser.add_argument("--desired_class", type=int, default=0)
args = parser.parse_args()

DESIRED_CLASS = args.desired_class
DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
LATENT_DIM = 16

class GaussianMutation(Mutation):

    def __init__(self, prob=0.9, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once

    def _do(self, problem, X, params=None, *args, random_state=None, **kwargs):
        X = X.astype(float)

        gaussian_noise = np.random.normal(0, 0.3, X.shape)
        Xp = X + gaussian_noise
        Xp = np.clip(Xp, problem.xl, problem.xu)

        return Xp

class BaseProblem(Problem):
    def __init__(
        self,
        ensemble_model,
        vae,
        point_of_interest,
        min_x,
        max_x,
        n_features,
        n_obj,
        cost_func,
        desired_class
    ):
        super().__init__(
            n_var=n_features,
            n_obj=n_obj,
            n_ieq_constr=1,
            xl=np.repeat(math.floor(min_x), n_features),
            xu=np.repeat(math.floor(max_x), n_features),
        )
        self.model = ensemble_model
        self.vae = vae
        self.point_of_interest = point_of_interest
        self.cost_func = cost_func
        self.desired_class = desired_class

    def _evaluate(self, x, out, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")


class Model_Fixed_Decision_Deterministic(BaseProblem):
    def __init__(
        self, ensemble_model,vae, point_of_interest, min_x, max_x, n_features, cost_func, desired_class
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            vae=vae,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
            desired_class=desired_class
        )

    def _evaluate(self, x, out, *args, **kwargs):
        decoded = self.vae.decode(torch.Tensor(x).to(DEVICE)).cpu().detach().numpy()
        costs = self.cost_func(
            self.model, decoded, self.point_of_interest, desired_class=self.desired_class
        )
        f3 = costs[:, 2]  # Distance
        f4 = -costs[:, 3]  # Probability of the class
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f3, f4]
        out["G"] = [g1]


class Model_Fixed_Decision_Stochastic(BaseProblem):
    def __init__(
        self, ensemble_model,vae, point_of_interest, min_x, max_x, n_features, cost_func, desired_class
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            vae=vae,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
            desired_class=desired_class
        )

    def _evaluate(self, x, out, *args, **kwargs):
        decoded = self.vae.decode(torch.Tensor(x).to(DEVICE)).cpu().detach().numpy()
        costs = self.cost_func(
            self.model, decoded, self.point_of_interest, desired_class=self.desired_class
        )
        f1 = costs[:, 0]  # AU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f1, f3]
        out["G"] = [g1]


class Model_Stochastic_Decision_Fixed(BaseProblem):
    def __init__(
        self, ensemble_model, vae, point_of_interest, min_x, max_x, n_features, cost_func, desired_class
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            vae = vae,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
            desired_class=desired_class
        )


    def _evaluate(self, x, out, *args, **kwargs):
        decoded = self.vae.decode(torch.Tensor(x).to(DEVICE)).cpu().detach().numpy()
        costs = self.cost_func(
            self.model, decoded, self.point_of_interest, desired_class=self.desired_class
        )
        f2 = costs[:, 1]  # EU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f2, f3]
        out["G"] = [g1]


class Model_Stochastic_Decision_Stochastic(BaseProblem):
    def __init__(
        self, ensemble_model, vae, point_of_interest, min_x, max_x, n_features, cost_func, desired_class
    ):
        super().__init__(
            ensemble_model=ensemble_model,
            vae=vae,
            point_of_interest=point_of_interest,
            min_x=min_x,
            max_x=max_x,
            n_features=n_features,
            n_obj=2,
            cost_func=cost_func,
            desired_class=desired_class
        )


    def _evaluate(self, x, out, *args, **kwargs):
        decoded = self.vae.decode(torch.Tensor(x).to(DEVICE)).cpu().detach().numpy()
        costs = self.cost_func(
            self.model, decoded, self.point_of_interest, desired_class=self.desired_class
        )
        f1 = costs[:, 0] + costs[:, 1]  # TU
        f3 = costs[:, 2]  # Distance
        g1 = 0.51 - costs[:, 3]  # Some constraint

        out["F"] = [f1, f3]
        out["G"] = [g1]



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

def load_epistemic_model():
    lenet_model = LeNet_MNIST(drop_prob=0.5, in_channels=1, image_width=28, image_heigth=28)
    ensemble_lenet = Ensemble_Classifier(
        base_model=lenet_model, n_models=20, random_state=42
    )
    ensemble_lenet.load("models/MNIST/Ensemble_Classification/")
    ensemble_lenet.compile(backend=BACKEND)
    ensemble_lenet.to(
        DEVICE
    )
    return ensemble_lenet

def load_vae(latent_dim):
    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(
        torch.load(f"vae_artifacts_{latent_dim}/vae.pth",weights_only=True, map_location=DEVICE)
    )
    vae.compile(backend=BACKEND)
    vae.to(DEVICE)
    vae.eval()
    return vae

def load_point_of_interest(model,testloader):
    iter_test = iter(testloader)
    all_test_images, all_test_labels = [], []
    for image_batch, label_batch in iter_test:
        all_test_images.append(image_batch)
        all_test_labels.append(label_batch)
    all_test_images = torch.cat(all_test_images, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    for idx, image in enumerate(all_test_images):
        if image.shape != (1, 28, 28):
            raise ValueError(
                "Image shape is not correct. Expected (1, 28, 28), got {}".format(
                    image.shape
                )
            )
        prob = model.predict(image.view(1, *image.shape))[0].max()
        if prob < 1:
            print("Image with low probability: ", prob.item())
            print("Image shape: ", image.shape)
            break
        
    # Save image
    plt.imshow(image.squeeze().numpy().reshape(28,28), cmap="gray")
    plt.axis("off")
    plt.savefig(f"moo_mnist/{DESIRED_CLASS}/point_of_interest_{idx}.png", bbox_inches="tight")
    plt.close()

    point_of_interest = image.flatten(start_dim=1)
    return point_of_interest

def get_costs_mnist(epiuc_ensemble_model, X, point_of_interest, desired_class=DESIRED_CLASS):
    """
    Calculate the costs for each point in X with respect to the point of interest.
    :param X: The data points
    :param point_of_interest: The point of interest
    :return: An array of costs
    """
    probabilities, uncertainties = epiuc_ensemble_model.predict(
        X, raw_output=True
    )
    tu, au, eu = torch.tensor_split(uncertainties.T, 3, dim=0)
    probabilities = (
        probabilities.mean(dim=1).detach().numpy()[:, desired_class]
    )  # Convert to log2 probabilities
    distances = np.linalg.norm(X.reshape(-1,point_of_interest.shape[1]) - point_of_interest, axis=1)
    return np.vstack(
        [au.detach().numpy(), eu.detach().numpy(), distances / 28.0, probabilities]
    ).T

def get_algorithm_setup(vae, point_of_interest,n_obj,population_size=200):
    ref_img = point_of_interest.reshape(1,28,28).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        z_ref, _ = vae.encode(ref_img.to(DEVICE))  # [1, latent_dim]

    # Generate population by perturbing z_ref
    sigma = 0.1  # small pertubation of original image: 0.1 better than 0.2 better than 0.3
    print("-----STARTING SAMPLING------")
    print("POPULATION SIZE", population_size)
    print("SIGMA: ", sigma)
    population = z_ref.cpu() + sigma * torch.randn(population_size, LATENT_DIM)

    # Optionally clip to ±3
    population = torch.clamp(population, -3, 3).numpy()
    n_offsprings = 200
    sampling = population
    
    print("CROSSOVER: ")
    crossover_prob = 1
    crossover_prob_var = 1
    crossover_eta = 1
    crossover = SBX(prob=crossover_prob, prob_var=crossover_prob_var, eta=crossover_eta)
    print("\t Method: ", crossover)
    print("\t prob: ", crossover_prob)
    print("\t prob_var: ", crossover_prob_var)
    print("\t eta: ", crossover_eta)

    print("MUTATION: ")
    mutation_prob = 1
    mutation_eta = 1
    mutation = PM(prob=mutation_prob, eta=mutation_eta)
    print("\t Method: ", mutation)
    print("\t prob: ", mutation_prob)
    print("\t eta: ", mutation_eta)
    
    eliminate_duplicates=True
    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=n_offsprings,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=eliminate_duplicates,
    )
    # algorithm = NSGA3(
    #     pop_size=population_size,
    #     ref_dirs=get_reference_directions("das-dennis", n_obj, n_partitions=24),
    #     n_offsprings=n_offsprings,
    #     sampling=sampling,
    #     crossover=crossover,
    #     mutation=mutation,
    #     eliminate_duplicates=eliminate_duplicates,
    # )
    print("ALGORITHM: ", algorithm)
    print("\t pop_size", population_size)
    print("\t n_offsprings", n_offsprings)
    print("\t sampling", sampling)
    print("\t crossover", crossover)
    print("\t mutation", mutation)
    print("\t eliminate_duplicates", eliminate_duplicates)
    max_gen = 400
    termination = get_termination("n_gen", max_gen)
    print("TERMINATION: ", termination)
    print("\t max_gen", max_gen)
    print("____FINISHED SAMPLING____")

    return algorithm, termination

def get_pareto_front(pymoo_problem, algorithm, termination=None):
    if termination is None:
        res = minimize(
            pymoo_problem, algorithm, seed=1, save_history=False, verbose=True
        )
    else:
        res = minimize(
            pymoo_problem, algorithm, termination=termination, seed=1, save_history=False, verbose=True
        )

    return res.X, res.F

def load_pymoo_problems(epiuc_model, vae, point_of_interest, min_val, max_val, cost_func,n_features, desired_class):
    m_f_d_d = Model_Fixed_Decision_Deterministic(
        epiuc_model,
        vae,
        point_of_interest.numpy(),
        min_val,
        max_val,
        n_features,
        cost_func,
        desired_class
    )
    m_f_d_s = Model_Fixed_Decision_Stochastic(
        epiuc_model,
        vae,
        point_of_interest.numpy(),
        min_val,
        max_val,
        n_features,
        cost_func,
        desired_class
    )
    m_s_d_f = Model_Stochastic_Decision_Fixed(
        epiuc_model,
        vae,
        point_of_interest.numpy(),
        min_val,
        max_val,
        n_features,
        cost_func,
        desired_class
    )
    m_s_d_s = Model_Stochastic_Decision_Stochastic(
        epiuc_model,
        vae,
        point_of_interest.numpy(),
        min_val,
        max_val,
        n_features,
        cost_func,
        desired_class
    )
    return m_f_d_d, m_f_d_s, m_s_d_f, m_s_d_s

def visualize_pareto_point(vae, pareto_points, objectives, name, latent_dim=16):
    pareto_points = vae.decode(
        torch.Tensor(pareto_points).to(DEVICE)
    ).cpu().detach().numpy()
    
    if pareto_points is None or len(pareto_points) == 0:
        warning.warn("No Pareto points to visualize.")
        return

    n_points = pareto_points.shape[0]
    n_cols = min(5, n_points)  # Fix number of columns (e.g., 5)
    n_rows = int(np.ceil(n_points / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

    for i in range(n_points):
        axes[i].imshow(pareto_points[i].reshape(28, 28), cmap="gray")
        axes[i].axis("off")
        # Show all objectives for this point, formatted nicely
        if n_points == 1:
            obj_str = objectives[0]
        else:
            obj_str = " ".join([f"{obj:.2f}" for obj in objectives[i]])
        axes[i].set_title(f"Obj: {obj_str}", fontsize=10)

    # Hide unused axes
    for j in range(n_points, n_rows * n_cols):
        axes[j].axis("off")

    plt.suptitle(f"Pareto Points: {name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"moo_mnist/{DESIRED_CLASS}/pareto_points_{name}_{latent_dim}.png")



if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    print("-----STARTING COUNTERFACTUAL GENERATION------")
    print("DESIRED_CLASS: ", DESIRED_CLASS)
    print("LATENT_DIM: ", LATENT_DIM)
    os.makedirs(f"moo_mnist/{DESIRED_CLASS}", exist_ok=True)
    trainloader, testloader = load_mnist()
    epiuc_ensemble = load_epistemic_model()
    vae = load_vae(LATENT_DIM)
    point_of_interest = load_point_of_interest(epiuc_ensemble, testloader)
    
    min_val, max_val = -3, 3
    print("SEARCH SPACE: ", min_val, max_val)
    m_f_d_d, m_f_d_s, m_s_d_f, m_s_d_s = load_pymoo_problems(epiuc_ensemble, vae, point_of_interest, min_val, max_val, get_costs_mnist, LATENT_DIM, DESIRED_CLASS)

    for name,problem in {"M_F_D_D": m_f_d_d, "M_F_D_S": m_f_d_s, "M_S_D_F": m_s_d_f, "M_S_D_S": m_s_d_s}.items():
        algorithm,termination = get_algorithm_setup(vae, point_of_interest,n_obj=problem.n_obj, population_size=400)
        points,objectives = get_pareto_front(problem, algorithm, termination=termination)
        if points is None:
            warning.warn(f"No Pareto points found for {name}.")
            continue
        visualize_pareto_point(vae,points, objectives, name, latent_dim=LATENT_DIM)
        del points

    print("Pareto fronts visualized.")
