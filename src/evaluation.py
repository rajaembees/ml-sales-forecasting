import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

IND_VARS1 = ["Advertising Expenditure"]
IND_VARS2 = ["Advertising Expenditure", "Discount Percentage", "Product Price"]

MODEL_NAMES = [
    "Simple LR  (Adv. Expenditure only)",
    "Multiple LR  (Adv. Exp. + Discount % + Price)",
    "Multiple LR  (All Features)",
]


def _metrics(model, X, y):
    pred = model.predict(X)
    return pd.DataFrame(
        {
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "MAE": mean_absolute_error(y, pred),
            "MAPE (%)": mean_absolute_percentage_error(y, pred) * 100,
        },
        index=[0],
    )


def evaluate_models(models, X_train, X_test, y_train, y_test, output_dir="outputs"):
    lin_reg1, lin_reg2, lin_reg3 = models
    os.makedirs(output_dir, exist_ok=True)

    train_rows = [
        _metrics(lin_reg1, X_train[IND_VARS1], y_train),
        _metrics(lin_reg2, X_train[IND_VARS2], y_train),
        _metrics(lin_reg3, X_train, y_train),
    ]
    test_rows = [
        _metrics(lin_reg1, X_test[IND_VARS1], y_test),
        _metrics(lin_reg2, X_test[IND_VARS2], y_test),
        _metrics(lin_reg3, X_test, y_test),
    ]

    train_df = pd.concat(train_rows, ignore_index=True)
    train_df.index = MODEL_NAMES
    test_df = pd.concat(test_rows, ignore_index=True)
    test_df.index = MODEL_NAMES

    print("\n--- Training Performance ---")
    print(train_df.to_string(float_format="%.4f"))
    print("\n--- Test Performance ---")
    print(test_df.to_string(float_format="%.4f"))

    train_df.to_csv(os.path.join(output_dir, "train_performance.csv"))
    test_df.to_csv(os.path.join(output_dir, "test_performance.csv"))

    _save_comparison_chart(test_df, output_dir)
    _save_residual_plot(lin_reg3, X_test, y_test, output_dir)

    best = test_df["RMSE"].idxmin()
    print(f"\n--- Best Model: {best} ---")
    print(f"  RMSE : ${test_df.loc[best, 'RMSE']:,.2f}")
    print(f"  MAE  : ${test_df.loc[best, 'MAE']:,.2f}")
    print(f"  MAPE : {test_df.loc[best, 'MAPE (%)']:.2f}%")
    print(f"\nCSV results saved to {output_dir}/")
    return train_df, test_df


def _save_comparison_chart(test_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = ["steelblue", "darkorange", "seagreen"]
    for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE (%)"]):
        bars = ax.bar(range(len(MODEL_NAMES)), test_df[metric], color=colors)
        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels([f"M{i+1}" for i in range(len(MODEL_NAMES))])
        ax.set_title(metric)
        for bar, val in zip(bars, test_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8,
            )
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(
        legend_patches,
        [f"M{i+1}: {n}" for i, n in enumerate(MODEL_NAMES)],
        loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.22), fontsize=8,
    )
    plt.suptitle("Test-Set Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=100, bbox_inches="tight")
    plt.close()


def _save_residual_plot(model, X_test, y_test, output_dir):
    pred = model.predict(X_test)
    residuals = y_test.values - pred
    plt.figure(figsize=(7, 5))
    plt.scatter(pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linewidth=1.5, linestyle="--")
    plt.xlabel("Predicted Sales ($)")
    plt.ylabel("Residuals ($)")
    plt.title("Residual Plot - Best Model (All Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_plot.png"), dpi=100, bbox_inches="tight")
    plt.close()
