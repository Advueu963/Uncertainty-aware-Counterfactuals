from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from uncertainty_cfs.tabular_util import get_dataset


if __name__ == "__main__":
    DATA_FILES = [
        "bank",
        "churn",
        "compas",
        "diabetes",
        "fico",
        "home",
        "titanic",
        "breast_cancer",
        "boston_housing",
    ]

    for dataset_name in DATA_FILES:
        X_train, X_test, y_train, y_test = get_dataset(dataset_name)
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        print(f"Visualizing dataset: {dataset_name} with shape {X.shape}")
        if X.shape[1] > 50:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=X.shape[1] // 4, random_state=42)
            X = pca.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)
        df = pd.DataFrame()
        df["TSNE-1"] = X_embedded[:, 0]
        df["TSNE-2"] = X_embedded[:, 1]
        df["Label"] = y
        df["is_test"] = [
            "Test" if i >= len(X_train) else "Train" for i in range(len(X))
        ]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="TSNE-1",
            y="TSNE-2",
            hue="Label",
            style="is_test",
            palette="deep",
        )
        plt.title(f"t-SNE Visualization of {dataset_name} Dataset")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_tsne_visualization.png")
        plt.close()
