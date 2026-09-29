import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    results = {
        "extra_trees_domain": {
            "mae": 2.4626,
            "rmse": 3.0263,
            "r2": 0.5517,
        },
        "extra_trees_total": {
            "mae": 2.5041,
            "rmse": 3.0729,
            "r2": 0.5378,
        },
        "random_forest": {
            "mae": 2.6141,
            "rmse": 3.2203,
            "r2": 0.4924,
        },

        "elastic_net": {
            "mae": 2.9299,
            "rmse": 3.9460,
            "r2": 0.2759,
        },
        "hist_gradient_boosting": {
            "mae": 2.7849,
            "rmse": 3.4563,
            "r2": 0.4153,
        },
    }
    """
    "ridge": {
            "mae": 4.5747,
            "rmse": 27.6239,
            "r2": -36.3527,
        },
    """

    models = list(results.keys())

    mae = [results[m]["mae"] for m in models]
    rmse = [results[m]["rmse"] for m in models]
    r2 = [results[m]["r2"] for m in models]

    x = np.arange(len(models))
    width = 0.22

    plt.figure(figsize=(12, 6))

    plt.bar(x - width - 0.03, mae, width, label="MAE")
    plt.bar(x, rmse, width, label="RMSE")
    plt.bar(x + width + 0.03, r2, width, label="R2")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Metric value")

    plt.title("Regression Model Performance Comparison")

    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig("regression_model_comparison.png")

    plt.show()
