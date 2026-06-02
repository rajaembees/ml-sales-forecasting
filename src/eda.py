import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


POPULARITY_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]


def run_eda(data: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    os.makedirs(output_dir, exist_ok=True)
    _plot_histograms(data, output_dir)
    _plot_boxplots(data, output_dir)
    _plot_categorical_counts(data, output_dir)
    _plot_correlation_heatmap(data, output_dir)
    _plot_sales_by_popularity(data, output_dir)
    _plot_sales_by_region(data, output_dir)
    print(f"EDA plots saved to {output_dir}/")


def _plot_histograms(data, output_dir):
    num_cols = data.select_dtypes(include="number").columns.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(col, fontsize=9)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograms.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _plot_boxplots(data, output_dir):
    num_cols = data.select_dtypes(include="number").columns.tolist()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(col, fontsize=9)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Outlier Detection (Box Plots)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _plot_categorical_counts(data, output_dir):
    cat_cols = data.select_dtypes(exclude="number").columns.tolist()
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(10, 5))
    if len(cat_cols) == 1:
        axes = [axes]
    for i, col in enumerate(cat_cols):
        order = POPULARITY_ORDER if col == "Popularity" else None
        sns.countplot(data=data, x=col, order=order, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle("Categorical Feature Counts", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "categorical_counts.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _plot_correlation_heatmap(data, output_dir):
    plt.figure(figsize=(10, 7))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _plot_sales_by_popularity(data, output_dir):
    valid_order = [p for p in POPULARITY_ORDER if p in data["Popularity"].unique()]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, y="Sales", x="Popularity", order=valid_order)
    plt.title("Sales by Popularity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sales_by_popularity.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _plot_sales_by_region(data, output_dir):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=data, y="Sales", x="Region")
    plt.title("Sales by Region")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sales_by_region.png"), dpi=100, bbox_inches="tight")
    plt.close()
