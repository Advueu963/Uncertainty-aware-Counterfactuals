import torch
import numpy as np
from torch.distributions import Normal


def generate_moon(
    n_samples=500, radius=1.0, width=0.3, distance=0.1, noise=0.05, flip=False
):
    angles = np.linspace(0, np.pi, n_samples)
    r = radius + width * (np.random.rand(n_samples) - 0.5)
    x = r * np.cos(angles)
    y = r * np.sin(angles)

    if flip:
        x = -x + radius  # flip and shift right
        y = -y + distance  # shift down

    x += np.random.normal(0, noise, n_samples)
    y += np.random.normal(0, noise, n_samples)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return torch.column_stack([x, y])


def generate_x_points(n_samples=1000, bias=1):
    """
    Generate n_samples points in the simplex.
    """
    # Generate random points in the simplex
    x1_corr = Normal(0, 1).sample((n_samples,))
    x2_corr = Normal(0, 1).sample((n_samples,))
    points = torch.stack([x1_corr, x2_corr], dim=1) + bias
    return points


def load_one_moon(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    # Generate random points in the simplex
    # Generate both moons
    moon1 = generate_moon(n_samples)
    points = torch.concatenate([moon1], dim=0)
    probs = 1 / (1 + torch.exp(-points[:, 0]))  # sharper boundary with higher factor
    y_probs = torch.stack([1 - probs, probs], dim=-1)
    y_labels = torch.argmax(y_probs, dim=1)

    return points, y_labels, y_probs


def load_two_moon(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    # Generate random points in the simplex
    # Generate both moons
    moon1 = generate_moon(n_samples)
    moon2 = generate_moon(n_samples, flip=True)
    points = torch.concatenate([moon1, moon2], dim=0)

    y_labels = torch.cat(
        [
            torch.zeros(n_samples, dtype=torch.long),  # Moon 1
            torch.ones(n_samples, dtype=torch.long),  # Moon 2
        ]
    )

    return points, y_labels, None


def load_four_moon(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    # Generate random points in the simplex
    # Generate both moons
    moon1 = generate_moon(n_samples)
    moon2 = generate_moon(n_samples, flip=True)
    moon3 = generate_moon(n_samples) - torch.tensor([2, 0])  # Shifted left
    moon4 = generate_moon(n_samples, flip=True) - torch.tensor(
        [2, 0]
    )  # Shifted left and flipped
    points = torch.concatenate([moon1, moon2, moon3, moon4], dim=0)

    y_labels = torch.cat(
        [
            torch.zeros(n_samples, dtype=torch.long),  # Moon 1
            torch.ones(n_samples, dtype=torch.long),  # Moon 2
            2 * torch.ones(n_samples, dtype=torch.long),  # Moon 3
            3 * torch.ones(n_samples, dtype=torch.long),  # Moon 4
        ]
    )

    return points, y_labels, None


def load_bubbles(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    point_bunch = generate_x_points(n_samples=n_samples, bias=-4)
    point_bunch2 = generate_x_points(n_samples=n_samples, bias=4)
    points = torch.vstack((point_bunch, point_bunch2))
    probs = 1 / (1 + torch.exp(-points.sum(dim=1)))
    y_probs = torch.stack([probs, 1 - probs], dim=-1)
    y_labels = torch.argmax(y_probs, dim=1)

    return points, y_labels, y_probs


def load_bubbles_multiclass(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    point_bunch = generate_x_points(n_samples=n_samples, bias=np.array([-4, -4]))
    labels_bunch = torch.ones(n_samples, dtype=torch.long)
    point_bunch2 = generate_x_points(n_samples=n_samples, bias=np.array([4, 4]))
    labels_bunch2 = torch.zeros(n_samples, dtype=torch.long)
    point_bunch3 = generate_x_points(n_samples=n_samples, bias=np.array([-4, 4]))
    labels_bunch3 = torch.ones(n_samples, dtype=torch.long) + 1
    point_bunch4 = generate_x_points(n_samples=n_samples, bias=np.array([4, -4]))
    labels_bunch4 = torch.ones(n_samples, dtype=torch.long) + 2
    points = torch.vstack(
        (point_bunch, point_bunch2, point_bunch3, point_bunch4)
    ).float()
    y_labels = torch.cat([labels_bunch, labels_bunch2, labels_bunch3, labels_bunch4])

    return points, y_labels, None


def load_bubbles_noisy(n_samples=1000):
    """
    Generate n_samples points in the simplex.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    point_bunch = generate_x_points(n_samples=n_samples, bias=1)
    point_bunch2 = generate_x_points(n_samples=n_samples, bias=-1)

    point_bunch3 = generate_x_points(n_samples=n_samples, bias=12)

    points = torch.vstack((point_bunch, point_bunch2))
    probs = 1 / (1 + torch.exp(-points[:, 0]))

    prob2 = 1 / (1 + torch.exp(-point_bunch3[:, 0] + 20))

    y_probs1 = torch.stack([probs, 1 - probs], dim=-1)
    y_probs2 = torch.stack([prob2, 1 - prob2], dim=-1)

    y_probs = torch.vstack((y_probs1, y_probs2))
    points = torch.vstack((points, point_bunch3))
    y_labels = torch.argmax(y_probs, dim=1)

    return points, y_labels, y_probs


def load_breast_dataset(n_samples=1000):
    """
    Load the breast cancer dataset from sklearn and return it as a PyTorch tensor.
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long)

    return X, y, None


def load_breast_dataset_poi(start_class=0):
    """
    Load the breast cancer dataset from sklearn and return it as a PyTorch tensor.
    """
    np.random.seed(42)

    X, y, _ = load_breast_dataset()
    X_startclass = X[y == start_class]
    idx_cfs = np.random.choice(X_startclass.shape[0], size=100, replace=False)
    X_cfs = X_startclass[idx_cfs]
    return X_cfs


def load_ring_dataset(n_samples=1000, inner_radius=1.0, outer_radius=2.0, noise=0.1):
    """
    Generate a synthetic dataset with class 0 in the center and class 1 in a surrounding ring.

    Parameters:
        n_samples (int): Total number of samples.
        inner_radius (float): Radius threshold for class 0.
        outer_radius (float): Outer boundary of the data.
        noise (float): Standard deviation of Gaussian noise added to points.
        plot (bool): Whether to plot the generated dataset.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, 2).
        y (ndarray): Labels of shape (n_samples,).
    """
    n0 = n_samples // 2  # Class 0
    n1 = n_samples - n0  # Class 1

    # Class 0: Points within the inner radius
    r0 = inner_radius * np.sqrt(np.random.rand(n0))
    theta0 = 2 * np.pi * np.random.rand(n0)
    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    x0 += noise * np.random.randn(n0, 2)

    # Class 1: Points in the ring between inner_radius and outer_radius
    r1 = inner_radius + (outer_radius - inner_radius) * np.sqrt(np.random.rand(n1))
    theta1 = 2 * np.pi * np.random.rand(n1)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    x1 += noise * np.random.randn(n1, 2)

    # Combine
    X = torch.from_numpy(np.vstack([x0, x1]).astype(np.float32))

    center = np.array([0.0, 0.0])

    # Compute Euclidean distance from center
    distances = np.linalg.norm(X - center, axis=1, ord=2)

    # Normalize distance to [0, 1] within the transition region
    normalized_r = (distances - inner_radius) / (outer_radius - inner_radius)

    # Sigmoid-based transition
    p1 = 1 / (1 + np.exp(-(normalized_r - 0.5)))
    p0 = 1 - p1

    y_probs = torch.from_numpy(np.stack([p1, p0], axis=-1))
    y_labels = torch.argmax(y_probs, dim=1)

    return X, y_labels, y_probs


def load_ring_dataset_multiclass(n_samples=1000, noise=0.1):
    """
    Generate a synthetic dataset with class 0 in the center and class 1 in a surrounding ring.

    Parameters:
        n_samples (int): Total number of samples.
        inner_radius (float): Radius threshold for class 0.
        outer_radius (float): Outer boundary of the data.
        noise (float): Standard deviation of Gaussian noise added to points.
        plot (bool): Whether to plot the generated dataset.

    Returns:
        X (ndarray): Feature matrix of shape (n_samples, 2).
        y (ndarray): Labels of shape (n_samples,).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    inner_radius = 4.0

    # Class 0: Points within the inner radius
    r0 = inner_radius * np.sqrt(np.random.rand(n_samples))
    theta0 = 2 * np.pi * np.random.rand(n_samples)
    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    x0 += noise * np.random.randn(n_samples, 2)

    # Combine
    X = torch.from_numpy(np.vstack([x0]).astype(np.float32))

    center = np.array([0.0, 0.0])
    # Compute Euclidean distance from center
    distances = np.linalg.norm(X - center, axis=1, ord=2)
    # Normalize distance to [0, 1] within the transition region
    normalized_r = (distances - 0) / (inner_radius)
    # Create four segments of distances
    segments = np.array(
        [
            0.0,  # Class 0
            0.25,  # Class 1
            0.5,  # Class 2
            0.75,  # Class 3
        ]
    )
    y_labels = torch.zeros(n_samples, dtype=torch.long)

    y_class_1 = (normalized_r >= segments[0]) & (normalized_r < segments[1])
    y_labels += y_class_1
    y_class_2 = (normalized_r >= segments[1]) & (normalized_r < segments[2])
    y_labels += 2 * y_class_2
    y_class_3 = (normalized_r >= segments[2]) & (normalized_r < segments[3])
    y_labels += 3 * y_class_3

    return X.float(), y_labels, None


def load_l_dataset(n_samples=500):
    torch.manual_seed(42)
    np.random.seed(42)
    # Class 0:
    x0 = Normal(4, 4).sample((n_samples,))
    x1 = Normal(0, 0.5).sample((n_samples,))
    l_part_one = torch.stack([x0, x1], dim=-1)
    y_labels_part_one = (
        l_part_one[:, 0] >= 1.5
    ).long()  # Class 0 if x0 > 0, else Class 1
    # y_labels_part_one = torch.zeros(n_samples, dtype=torch.long)

    # Class 1:
    x0 = Normal(0, 0.5).sample((n_samples,))
    x1 = Normal(4, 4).sample((n_samples,))
    l_part_two = torch.stack([x0, x1], dim=-1)
    y_labels_part_two = (l_part_two[:, 1] < 0).long()  # Class 1 if x0 > 0, else Class 0
    # y_labels_part_two = torch.ones(n_samples, dtype=torch.long)

    # Class 1 point bunch
    point_bunch = generate_x_points(n_samples=n_samples // 5, bias=10)
    y_labels_bunch = torch.ones(n_samples // 5, dtype=torch.long)

    X = torch.vstack([l_part_one, l_part_two, point_bunch])
    y_labels = torch.cat([y_labels_part_one, y_labels_part_two, y_labels_bunch])

    return X, y_labels, None


def load_l_dataset_multiclass(n_samples=500):
    torch.manual_seed(42)
    np.random.seed(42)
    # Class 0:
    x0 = Normal(4, 4).sample((n_samples,))
    x1 = Normal(0, 0.5).sample((n_samples,))
    l_part_one = torch.stack([x0, x1], dim=-1)
    y_labels_part_one = (
        l_part_one[:, 0] >= 1.5
    ).long()  # Class 0 if x0 > 0, else Class 1
    # y_labels_part_one = torch.zeros(n_samples, dtype=torch.long)

    # Class 1:
    x0 = Normal(0, 0.5).sample((n_samples,))
    x1 = Normal(4, 4).sample((n_samples,))
    l_part_two = torch.stack([x0, x1], dim=-1)
    y_labels_part_two = (l_part_two[:, 1] < 0).long()  # Class 1 if x0 > 0, else Class 0
    # y_labels_part_two = torch.ones(n_samples, dtype=torch.long)

    # Class 2:
    x0 = Normal(-4, 4).sample((n_samples,))
    x1 = Normal(0, 0.5).sample((n_samples,))
    l_part_three = torch.stack([x0, x1], dim=-1)
    y_labels_part_three = (
        2 * (l_part_three[:, 0] < 0).long()
    )  # Class 0 if x0 > 0, else Class 1

    # Class 3:
    x0 = Normal(0, 0.5).sample((n_samples,))
    x1 = Normal(-4, 4).sample((n_samples,))
    l_part_four = torch.stack([x0, x1], dim=-1)
    y_labels_part_four = (
        3 * (l_part_four[:, 1] < 0).long()
    )  # Class 0 if x0 > 0, else Class 1

    # Class 1 point bunch
    point_bunch = generate_x_points(n_samples=n_samples // 5, bias=np.array([10, 10]))
    y_labels_bunch = torch.ones(n_samples // 5, dtype=torch.long)

    # Class 0 point bunch
    point_bunch_2 = generate_x_points(
        n_samples=n_samples // 5, bias=np.array([-10, 10])
    )
    y_labels_bunch_2 = torch.zeros(n_samples // 5, dtype=torch.long)

    point_bunch_4 = generate_x_points(
        n_samples=n_samples // 5, bias=np.array([-10, -10])
    )
    y_labels_bunch_4 = 2 * torch.ones(n_samples // 5, dtype=torch.long)

    point_bunch_3 = generate_x_points(
        n_samples=n_samples // 5, bias=np.array([10, -10])
    )
    y_labels_bunch_3 = 3 * torch.ones(n_samples // 5, dtype=torch.long)

    X = torch.vstack(
        [
            l_part_one,
            l_part_two,
            l_part_three,
            l_part_four,
            point_bunch,
            point_bunch_2,
            point_bunch_3,
            point_bunch_4,
        ]
    )
    y_labels = torch.cat(
        [
            y_labels_part_one,
            y_labels_part_two,
            y_labels_part_three,
            y_labels_part_four,
            y_labels_bunch,
            y_labels_bunch_2,
            y_labels_bunch_3,
            y_labels_bunch_4,
        ]
    )

    return X.float(), y_labels, None


def load_infinity_dataset(n_samples=1000):
    n0 = n_samples // 2
    n1 = n_samples - n0

    r0 = (np.random.rand(n0)) * 2 * np.pi
    x0 = np.stack([np.cos(r0), np.sin(r0)], axis=1) - np.array([1, 0])

    r1 = (np.random.rand(n1)) * 2 * np.pi
    x1 = np.stack([np.cos(r1), np.sin(r1)], axis=1) + np.array([1, 0])

    X = torch.from_numpy(np.vstack([x0, x1]).astype(np.float32))

    y_probs = torch.zeros((n_samples, 2))
    y_probs[:n0, 1] = 1.0
    y_probs[n0:, 0] = 1.0

    y_labels = torch.argmax(y_probs, dim=1)

    return X, y_labels, y_probs


def load_datasets(dataset_name, noisy=False, **kwargs):
    if dataset_name == "one_moon":
        X, y_labels, y_probs = load_one_moon(**kwargs)
    elif dataset_name == "two_moon":
        X, y_labels, y_probs = load_two_moon(**kwargs)
    elif dataset_name == "infinity":
        X, y_labels, y_probs = load_infinity_dataset(**kwargs)
    elif dataset_name == "ring_dataset":
        X, y_labels, y_probs = load_ring_dataset(**kwargs)
    elif dataset_name == "l_dataset":
        X, y_labels, y_probs = load_l_dataset(**kwargs)
    elif dataset_name == "bubbles":
        X, y_labels, y_probs = load_bubbles(**kwargs)
    elif dataset_name == "bubbles_noisy":
        X, y_labels, y_probs = load_bubbles_noisy(**kwargs)
    elif dataset_name == "infinity_dataset":
        X, y_labels, y_probs = load_infinity_dataset(**kwargs)
    elif dataset_name =="breast_cancer":
        X, y_labels, y_probs = load_breast_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if noisy:
        # Now extend 8 features that are just noise
        torch.manual_seed(42)
        noise_features = torch.distributions.Normal(0, 1).sample((X.shape[0], 8))
        X_extended = torch.cat((X, noise_features), dim=1)
        return X_extended, y_labels, y_probs
    return X, y_labels, y_probs
