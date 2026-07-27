from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = [
    {"model": "Dummy", "mae": 3.81, "rmse": 4.79, "r2": -0.02, "spearman_rho": None},
    {"model": "Ridge", "mae": 3.01, "rmse": 3.93, "r2": 0.315, "spearman_rho": 0.58},
    {"model": "Elastic Net", "mae": 3.01, "rmse": 3.83, "r2": 0.348, "spearman_rho": 0.57},
    {"model": "Random Forest", "mae": 2.8126, "rmse": 3.5268, "r2": 0.4477, "spearman_rho": 0.6197},
    {"model": "Extra Trees", "mae": 2.7747, "rmse": 3.4169, "r2": 0.4815, "spearman_rho": 0.6394},
    {"model": "HistGradientBoosting", "mae": 2.9822, "rmse": 3.7538, "r2": 0.3743, "spearman_rho": 0.55},
]

OUTPUT_DIR = Path(
    r"C:\Users\ROG\IHE-project\data\processed\qc_v3\v4"
    r"\grouped_cv\final_model_plots"
)


def add_value_labels(ax, values, decimals=2):
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        ax.annotate(
            f"{value:.{decimals}f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
        )


def make_bar_plot(df, metric, title, ylabel, filename,
                  ascending, decimals=2, ylim=None, zero_line=False):
    plot_df = df.dropna(subset=[metric]).sort_values(
        metric, ascending=ascending
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(plot_df["model"], plot_df[metric])

    if zero_line:
        ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    add_value_labels(ax, plot_df[metric], decimals)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(RESULTS)
    df.to_csv(
        OUTPUT_DIR / "model_metrics_summary.csv",
        index=False,
    )

    make_bar_plot(
        df,
        metric="mae",
        title="Grouped OOF Mean Absolute Error by Model",
        ylabel="MAE (M-GEARS points; lower is better)",
        filename="model_comparison_mae.png",
        ascending=True,
        decimals=2,
    )

    make_bar_plot(
        df,
        metric="rmse",
        title="Grouped OOF Root Mean Squared Error by Model",
        ylabel="RMSE (M-GEARS points; lower is better)",
        filename="model_comparison_rmse.png",
        ascending=True,
        decimals=2,
    )

    make_bar_plot(
        df,
        metric="r2",
        title="Grouped OOF R² by Model",
        ylabel="R² (higher is better)",
        filename="model_comparison_r2.png",
        ascending=False,
        decimals=3,
        zero_line=True,
    )

    make_bar_plot(
        df,
        metric="spearman_rho",
        title="Grouped OOF Spearman Correlation by Model",
        ylabel="Spearman ρ (higher is better)",
        filename="model_comparison_spearman.png",
        ascending=False,
        decimals=3,
        ylim=(0, 0.75),
    )

    print("Plots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()