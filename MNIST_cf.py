import torch
import torchvision
import torchvision.transforms as transforms

from epiuc.uncertainty.classification import LeNet_MNIST
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from property_procedures.utils import ensemble_probs, total_uncertainty_ensemble, epistemic_uncertainty_ensemble, \
    aleatoric_uncertainty_ensemble
from property_procedures import (
    validity_loss_function,
    connected_loss_function,
    robust_loss_function,
    feasable_loss_function,
    discriminative_loss_function,
    plausable_loss_function,
    similarity_loss_function,
    counter_factual_optimization_routine,
    counter_factual_optimization_routine_schut,
    combined_loss_function,
)
from config import DATALOADER_CONFIGS, BACKEND

DESIRED_CLASS = 6
MAX_STEPS = 5000
DELTA = 4
N_POINTS = 10
OPTIMIZER_LR = 0.2
PROB_WEIGHT = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
PATIENCE = MAX_STEPS
DESIRED_VALIDITY = 0.9
PROPERTY_LOADERS = [
    (
        "validity",
        validity_loss_function,
    ),
    (
        "connected_ball",
        connected_loss_function,
    ),
    (
        "robust",
        robust_loss_function,
    ),
    (
        "feasability",
        feasable_loss_function,
    ),
    (
        "discriminative",
        discriminative_loss_function,
    ),
    (
        "plausable",
        plausable_loss_function,
    ),
    (
        "similarity",
        similarity_loss_function,
    ),
    ("combined", combined_loss_function),
]


def load_mnist(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS, ROOT_PATH="../data"):
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
    trainloader = torch.utils.data.DataLoader(
        trainset, **loading_configs
    )

    testset = torchvision.datasets.MNIST(
        root=ROOT_PATH, train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=loading_configs["batch_size"],
        shuffle=False,
        num_workers=loading_configs["num_workers"],
        pin_memory=loading_configs["pin_memory"]
    )

    return trainloader, testloader
trainloader, testloader = load_mnist()
lenet_model = LeNet_MNIST(drop_prob=0.5, in_channels=1, image_width=28, image_heigth=28)
ensemble_lenet = Ensemble_Classifier(
    base_model=lenet_model,
    n_models=5,
    random_state=42
)
ensemble_lenet.load("models/Ensemble_MNIST/")
ensemble_lenet.compile(backend=BACKEND)
ensemble_lenet.to(device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"))

iter_test = iter(testloader)
all_test_images, all_test_labels = [], []
for image_batch, label_batch in iter_test:
    all_test_images.append(image_batch)
    all_test_labels.append(label_batch)
all_test_images = torch.cat(all_test_images, dim=0)
all_test_labels = torch.cat(all_test_labels, dim=0)
all_test_images.shape

for idx, image in enumerate(all_test_images):
    if image.shape != (1, 28, 28):
        raise ValueError("Image shape is not correct. Expected (1, 28, 28), got {}".format(image.shape))
    prob = ensemble_lenet.predict(image.view(1, *image.shape))[0].max()
    if prob < 1:
        print("Image with low probability: ", prob.item())
        print("Image shape: ", image.shape)
        break
    
# visualize the image
import matplotlib.pyplot as plt
plt.imshow(image.squeeze().numpy(), cmap="gray")
plt.savefig("mnist_image_original_{}.png".format(idx))
print("Label = ", all_test_labels[idx])

print("probability:", ensemble_lenet.predict(image.view(1, *image.shape),raw_output=False)[0])

#### Validity

point_of_interest = image.view(1, *image.shape)  # Reshape to (1, 1, 28, 28) for the model

for property_name, loss_function in PROPERTY_LOADERS:
    counter_factual, counter_factual_steps = counter_factual_optimization_routine_schut(
                point_to_explain=point_of_interest,
                model=ensemble_lenet,
                probability_function=ensemble_probs,
                desired_class=DESIRED_CLASS,
                loss_function=loss_function,
                aleatoric_uncertainty_function=aleatoric_uncertainty_ensemble,
                epistemic_uncertainty_function=epistemic_uncertainty_ensemble,
                MAX_STEPS=MAX_STEPS,
                delta=DELTA,
                n_points=N_POINTS,
                lr=OPTIMIZER_LR,
                DESIRED_VALIDITY=DESIRED_VALIDITY,
                p_weight=PROB_WEIGHT,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                patience=PATIENCE,
                optimization_method="sgd",
            )
    plt.imshow(counter_factual.squeeze().numpy(), cmap="gray")
    plt.savefig("mnist_image_counterfactual_{}_{}_2.png".format(property_name, idx))
