import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Feature subsets used across models
IND_VARS1 = ["Advertising Expenditure"]
IND_VARS2 = ["Advertising Expenditure", "Discount Percentage", "Product Price"]

POPULARITY_ENCODING = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}


def preprocess(data: pd.DataFrame):
    """Encode categoricals and return (X, y)."""
    X = data.drop("Sales", axis=1).copy()
    y = data["Sales"]

    X["Popularity"] = X["Popularity"].map(POPULARITY_ENCODING)

    X = pd.get_dummies(
        X,
        columns=X.select_dtypes(include=["object", "category"]).columns.tolist(),
        drop_first=True,
    )
    X = X.astype(float)
    return X, y


def build_models(data: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Build all three models and return them with the train/test split."""
    os.makedirs(output_dir, exist_ok=True)

    X, y = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Model 1 – Simple Linear Regression
    lin_reg1 = LinearRegression()
    lin_reg1.fit(X_train[IND_VARS1], y_train)
    _print_equation(lin_reg1, IND_VARS1, "Model 1 – Simple LR")
    _save_regression_line(lin_reg1, X_train[IND_VARS1], y_train, IND_VARS1[0], output_dir)

    # Model 2 – Multiple LR with 3 features
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X_train[IND_VARS2], y_train)
    _print_equation(lin_reg2, IND_VARS2, "Model 2 – Multiple LR (3 features)")

    # Model 3 – Multiple LR with all features
    lin_reg3 = LinearRegression()
    lin_reg3.fit(X_train, y_train)
    _print_equation(lin_reg3, X_train.columns.tolist(), "Model 3 – Multiple LR (all features)")

    return (lin_reg1, lin_reg2, lin_reg3), X_train, X_test, y_train, y_test


def _print_equation(model, feature_names, title):
    print(f"\n{title}")
    terms = " + ".join(f"({c:.4f})*{v}" for c, v in zip(model.coef_, feature_names))
    print(f"  Sales = {terms} + {model.intercept_:.4f}")


def _save_regression_line(model, X_train, y_train, feature_name, output_dir):
    fitted = model.predict(X_train)
    sorted_idx = X_train[feature_name].argsort()
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[feature_name], y_train, alpha=0.3, s=10, label="Training data")
    plt.plot(
        X_train[feature_name].iloc[sorted_idx],
        fitted[sorted_idx],
        color="royalblue",
        linewidth=2,
        label="Best-fit line",
    )
    plt.xlabel(feature_name)
    plt.ylabel("Sales ($)")
    plt.title("Simple Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simple_lr_fit.png"), dpi=100, bbox_inches="tight")
    plt.close()
