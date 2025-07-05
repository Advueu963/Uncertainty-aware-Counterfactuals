import torch
import torchvision
import torchvision.transforms as transforms
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from epiuc.uncertainty.classification import LeNet_MNIST
from epiuc.uncertainty.wrapper import Ensemble_Classifier
from property_procedures.utils import ensemble_probs, total_uncertainty_ensemble, epistemic_uncertainty_ensemble, \
    aleatoric_uncertainty_ensemble
from epiuc.utils.eval_metrics import evaluate_accuracy
    
from config import DATALOADER_CONFIGS, BACKEND


def load_mnist(vali_size=0.1, generator=None, loading_configs=DATALOADER_CONFIGS, ROOT_PATH="data"):
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
    n_models=20,
    random_state=42
)
ensemble_lenet.load("models/MNIST/Ensemble_Classification/")

print(evaluate_accuracy([ensemble_lenet], testloader))