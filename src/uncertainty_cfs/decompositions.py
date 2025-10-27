#### Code adapted from probly package ####
import torch


def total_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute the total entropy as the total uncertainty.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)
        base: float, default=2
    Returns:
        te: torch.Tensor of shape (n_instances,)

    """
    probs = torch.mean(probs, dim=1)
    te = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
    return te.clamp(min=1e-10, max=1 - 1e-10)


def conditional_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute conditional entropy as the aleatoric uncertainty.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)
        base: float, default=2
    Returns:
        ce: torch.Tensor of shape (n_instances,)

    """
    ce = -torch.sum(probs * torch.log2(probs + 1e-10), dim=2)
    ce = torch.mean(ce, dim=1)
    return ce.clamp(min=1e-10, max=1 - 1e-10)


def mutual_information(probs: torch.Tensor) -> torch.Tensor:
    """Compute the mutual information as epistemic uncertainty.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)
        base: float, default=2
    Returns:
        mi: torch.Tensor of shape (n_instances,)

    """
    probs_mean = torch.mean(probs, dim=1)
    probs_mean = torch.repeat_interleave(
        torch.unsqueeze(probs_mean, 1), repeats=probs.shape[1], dim=1
    )
    mi = torch.sum(
        probs * (torch.log2(probs + 1e-10) - torch.log2(probs_mean + 1e-10)), dim=2
    )
    mi = torch.mean(mi, dim=1)
    return mi.clamp(min=1e-10, max=1 - 1e-10)


def entropy_based_uncertainty_quantification(predictions: torch.Tensor) -> tuple:
    """Returns the entropy based uncertainty quantification metrics.

    Args:
        preds (torch.Tensor): The predicted probabilities for each class. Must be of shape (n_instances, n_samples, n_classes).

    Returns:
        tuple: A tuple containing the total entropy, aleatoric uncertainty, and epistemic uncertainty.
    """
    eu = mutual_information(predictions)
    au = conditional_entropy(predictions)
    te = total_entropy(predictions)
    return te, au, eu


def total_variance(probs: torch.Tensor) -> torch.Tensor:
    """Compute the total uncertainty using variance-based measures.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)

    Returns:
        tv: torch.Tensor, shape (n_instances,)

    """
    probs_mean = probs.mean(axis=1)
    tv = torch.sum(probs_mean * (1 - probs_mean), dim=1)
    return tv


def expected_conditional_variance(probs: torch.Tensor) -> torch.Tensor:
    """Compute the aleatoric uncertainty using variance-based measures.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)

    Returns:
        ecv: torch.Tensor, shape (n_instances,)

    """
    ecv = torch.sum(torch.mean(probs * (1 - probs), dim=1), dim=1)
    return ecv


def variance_conditional_expectation(probs: torch.Tensor) -> torch.Tensor:
    """Compute the epistemic uncertainty using variance-based measures.

    The computation is based on samples from a second-order distribution.

    Args:
        probs: torch.Tensor of shape (n_instances, n_samples, n_classes)

    Returns:
        ecv: torch.Tensor, shape (n_instances,)

    """
    probs_mean = torch.mean(probs, dim=1, keepdim=True)
    vce = torch.sum(torch.mean(probs * (probs - probs_mean), dim=1), dim=1)
    return vce.clamp(min=0.0)


def variance_based_uncertainty_quantification(preds: torch.Tensor) -> tuple:
    """Returns the entropy based uncertainty quantification metrics.

    Args:
        preds (torch.Tensor): The predicted probabilities for each class. Must be of shape (n_instances, n_samples, n_classes).

    Returns:
        tuple: A tuple containing the total entropy, aleatoric uncertainty, and epistemic uncertainty.
    """
    eu = variance_conditional_expectation(preds)
    au = expected_conditional_variance(preds)
    te = total_variance(preds)
    return te, au, eu
