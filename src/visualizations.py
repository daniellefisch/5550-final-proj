from __future__ import annotations
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# paths
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = BASE_DIR / "outputs"
VIZ_DIR = OUTPUTS_DIR / "visualizations"
DATA_PATH = BASE_DIR / "data" / "clean" / "final_model_data.csv"

VIZ_DIR.mkdir(parents=True, exist_ok=True)


# region definitions
CORN_BELT_STATES = {
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "MINNESOTA",
    "MISSOURI",
    "OHIO",
    "WISCONSIN",
}

GREAT_PLAINS_STATES = {
    "KANSAS",
    "NEBRASKA",
    "NORTH DAKOTA",
    "OKLAHOMA",
    "SOUTH DAKOTA",
    "TEXAS",
}


# helpers
def clean_model_name(x: str) -> str:
    mapping = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boost": "Gradient Boosting",
    }
    return mapping.get(x, x)


def clean_feature_set(x: str) -> str:
    mapping = {
        "climate_only": "Climate Only",
        "climate_plus_drought": "Climate + Drought",
    }
    return mapping.get(x, x)


def clean_eval_name(x: str) -> str:
    mapping = {
        "great_plains_transfer": "Corn Belt → Great Plains",
        "cb_to_gp_transfer": "Corn Belt → Great Plains",
        "gp_to_cb_transfer": "Great Plains → Corn Belt",
        "corn_belt_cv": "Corn Belt CV",
        "great_plains_cv": "Great Plains CV",
    }
    return mapping.get(x, x)


def save_fig(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    fig.savefig(VIZ_DIR / filename, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_value_labels(ax, bars, fmt="{:.2f}", fontsize=9, offset=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# read data
lin_metrics = pd.read_csv(OUTPUTS_DIR / "linear_regression_metrics.csv")
rf_metrics = pd.read_csv(OUTPUTS_DIR / "random_forest_metrics.csv")
gb_metrics = pd.read_csv(OUTPUTS_DIR / "gradient_boost_metrics.csv")
gp_extra = pd.read_csv(OUTPUTS_DIR / "great_plains_additional_metrics.csv")

rf_cb_to_gp_preds = pd.read_csv(OUTPUTS_DIR / "random_forest_cb_to_gp_transfer_predictions.csv")
rf_gp_to_cb_preds = pd.read_csv(OUTPUTS_DIR / "random_forest_gp_to_cb_transfer_predictions.csv")

gb_cb_to_gp_preds = pd.read_csv(OUTPUTS_DIR / "gradient_boost_transfer_predictions.csv")
gb_gp_to_cb_preds = pd.read_csv(OUTPUTS_DIR / "gradient_boost_gp_to_cb_transfer_predictions.csv")

fi_compare = pd.read_csv(OUTPUTS_DIR / "feature_importance_comparison.csv")
raw_df = pd.read_csv(DATA_PATH)


# combine metrics
metrics_all = pd.concat([lin_metrics, rf_metrics, gb_metrics, gp_extra], ignore_index=True)
metrics_all["model_clean"] = metrics_all["model"].apply(clean_model_name)
metrics_all["feature_set_clean"] = metrics_all["feature_set"].apply(clean_feature_set)
metrics_all["evaluation_set_clean"] = metrics_all["evaluation_set"].apply(clean_eval_name)


# FIGURE 1: RQ2 Transfer heatmap
transfer_df = metrics_all[
    (metrics_all["evaluation_set_clean"].isin(["Corn Belt → Great Plains", "Great Plains → Corn Belt"])) &
    (metrics_all["feature_set_clean"] == "Climate + Drought")
].copy()

model_order = ["Linear Regression", "Random Forest", "Gradient Boosting"]
eval_order = ["Corn Belt → Great Plains", "Great Plains → Corn Belt"]

heatmap_data = (
    transfer_df.pivot(index="model_clean", columns="evaluation_set_clean", values="r2")
    .reindex(index=model_order, columns=eval_order)
)

heatmap_cmap = LinearSegmentedColormap.from_list(
    "custom_red_white_blue",
    ["#b01111", "#f7f7f7", "#62a1db"]
)

norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
im = ax.imshow(
    heatmap_data.values,
    aspect="auto",
    cmap=heatmap_cmap,
    norm=norm
)
ax.set_xticks(np.arange(len(eval_order)))
ax.set_xticklabels(eval_order, fontsize=11)
ax.set_yticks(np.arange(len(model_order)))
ax.set_yticklabels(model_order, fontsize=11)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")

ax.set_title("Transfer Performance", fontsize=15, fontweight="bold")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("R²", rotation=270, labelpad=15)

save_fig(fig, "rq2_transfer_heatmap.png")


# FIGURE 2: RQ1 Within-region bar chart
within_df = metrics_all[
    (metrics_all["evaluation_set_clean"].isin(["Corn Belt CV", "Great Plains CV"])) &
    (metrics_all["feature_set_clean"] == "Climate + Drought")
].copy()

within_eval_order = ["Corn Belt CV", "Great Plains CV"]
within_df["evaluation_set_clean"] = pd.Categorical(
    within_df["evaluation_set_clean"], categories=within_eval_order, ordered=True
)
within_df["model_clean"] = pd.Categorical(
    within_df["model_clean"], categories=model_order, ordered=True
)
within_df = within_df.sort_values(["evaluation_set_clean", "model_clean"])

x = np.arange(len(within_eval_order))
width = 0.24

fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

colors = {
    "Linear Regression": "#b0b0b0",
    "Random Forest": "#5B2A86",
    "Gradient Boosting": "#5962de"
}

for ax, region in zip(axes, ["Corn Belt CV", "Great Plains CV"]):
    subset = within_df[within_df["evaluation_set_clean"] == region]

    vals = []
    for model in model_order:
        row = subset[subset["model_clean"] == model]
        vals.append(row["r2"].iloc[0] if not row.empty else np.nan)

    y_pos = np.arange(len(model_order))

    bars = ax.barh(
        y_pos,
        vals,
        color=[colors[m] for m in model_order],
        height=0.6
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_order)

    # Labels on bars
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)

    ax.set_title(region, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1].set_xlabel("R²", fontsize=12)

fig.suptitle("Within-Region Predictive Performance", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, "rq1_within_region_r2.png")


# FIGURE: RQ2 supporting plot

dist_df = raw_df.copy()
dist_df["region"] = np.where(
    dist_df["state"].isin(CORN_BELT_STATES),
    "Corn Belt",
    np.where(dist_df["state"].isin(GREAT_PLAINS_STATES), "Great Plains", np.nan),
)
dist_df = dist_df.dropna(subset=["region"]).copy()

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

plot_vars = {
    "prism_tmax_may_sep_mean": "Temperature",
    "prism_ppt_may_sep_total": "Precipitation",
    "drought_intensity_d2plus": "Drought Intensity",
}

region_colors = {
    "Corn Belt": "#5962de",
    "Great Plains": "#5B2A86",
}

for ax, (col, label) in zip(axes, plot_vars.items()):
    cb_vals = dist_df.loc[dist_df["region"] == "Corn Belt", col].dropna()
    gp_vals = dist_df.loc[dist_df["region"] == "Great Plains", col].dropna()

    bp = ax.boxplot(
        [cb_vals, gp_vals],
        tick_labels=["Corn Belt", "Great Plains"],
        patch_artist=True,
        widths=0.6,
        showfliers=False
    )

    bp["boxes"][0].set_facecolor(region_colors["Corn Belt"])
    bp["boxes"][1].set_facecolor(region_colors["Great Plains"])

    for box in bp["boxes"]:
        box.set_alpha(0.8)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Climate Stress Distributions Differ Across Regions", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])

save_fig(fig, "rq2_region_distribution_comparison.png")


# FIGURE 4: RQ3 Feature importance comparison
fi_df = fi_compare.copy()

feature_name_map = {
    "prism_tmax_may_sep_mean": "Temperature",
    "prism_ppt_may_sep_total": "Precipitation",
    "drought_intensity_d2plus": "Drought Intensity",
    "drought_freq_d2plus": "Drought Frequency",
}

fi_df["feature_clean"] = fi_df["feature"].map(feature_name_map).fillna(fi_df["feature"])
fi_df["trained_on_clean"] = fi_df["trained_on"].replace(
    {"corn_belt": "Corn Belt", "great_plains": "Great Plains"}
)
fi_df["evaluated_on_clean"] = fi_df["evaluated_on"].replace(
    {"corn_belt": "Corn Belt", "great_plains": "Great Plains"}
)

# keep within-region comparisons only
fi_df = fi_df[fi_df["trained_on_clean"] == fi_df["evaluated_on_clean"]].copy()

feature_order = ["Temperature", "Precipitation", "Drought Intensity", "Drought Frequency"]
region_order = ["Corn Belt", "Great Plains"]

x = np.arange(len(feature_order))
width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(8, 9))

region_colors = {
    "Corn Belt": "#5962de",
    "Great Plains": "#5B2A86"
}

regions = ["Corn Belt", "Great Plains"]

for ax, region in zip(axes, regions):
    subset = fi_df[fi_df["trained_on_clean"] == region].copy()

    subset["feature_clean"] = pd.Categorical(
        subset["feature_clean"],
        categories=feature_order,
        ordered=True
    )
    subset = subset.sort_values("feature_clean")

    x = np.arange(len(feature_order))

    bars = ax.bar(
        x,
        subset["importance_mean"],
        yerr=subset["importance_std"],
        capsize=4,
        color=region_colors[region],
        alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=20, ha="right")

    # Labels
    for i, v in enumerate(subset["importance_mean"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    ax.set_title(region, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1].set_ylabel("Permutation Importance")
fig.suptitle("Regional Differences in Climate Drivers", fontsize=15, fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, "rq3_feature_importance.png")


# FIGURE 5: Paper figure - distribution comparison
raw_df = raw_df.copy()
raw_df["region"] = np.where(
    raw_df["state"].isin(CORN_BELT_STATES),
    "Corn Belt",
    np.where(raw_df["state"].isin(GREAT_PLAINS_STATES), "Great Plains", np.nan),
)
raw_df = raw_df.dropna(subset=["region"]).copy()

dist_vars = {
    "prism_tmax_may_sep_mean": "Temperature",
    "prism_ppt_may_sep_total": "Precipitation",
    "drought_intensity_d2plus": "Drought Intensity",
}

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))

for ax, (col, label) in zip(axes, dist_vars.items()):
    cb_vals = raw_df.loc[raw_df["region"] == "Corn Belt", col].dropna()
    gp_vals = raw_df.loc[raw_df["region"] == "Great Plains", col].dropna()

    ax.boxplot([cb_vals, gp_vals], tick_labels=["Corn Belt", "Great Plains"])
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Climate Stress Distributions Differ Across Regions", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

save_fig(fig, "paper_region_distribution_comparison.png")


# FIGURE 6: Paper figure - RF error vs drought
rf_error_df = pd.concat(
    [
        rf_cb_to_gp_preds.assign(direction="Corn Belt → Great Plains").rename(
            columns={"predicted_yield_climate_plus_drought": "predicted"}
        ),
        rf_gp_to_cb_preds.assign(direction="Great Plains → Corn Belt").rename(
            columns={"predicted_yield_climate_plus_drought": "predicted"}
        ),
    ],
    ignore_index=True,
)

rf_error_df = rf_error_df.merge(
    raw_df[["fips", "state", "county", "year", "drought_intensity_d2plus", "prism_ppt_may_sep_total"]],
    on=["fips", "state", "county", "year"],
    how="left",
)

rf_error_df["abs_error"] = np.abs(rf_error_df["yield_bu_acre"] - rf_error_df["predicted"])

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)

for ax, direction in zip(axes, ["Corn Belt → Great Plains", "Great Plains → Corn Belt"]):
    subset = rf_error_df[rf_error_df["direction"] == direction]
    ax.scatter(subset["drought_intensity_d2plus"], subset["abs_error"], alpha=0.25, s=12)
    ax.set_title(direction, fontsize=12, fontweight="bold")
    ax.set_xlabel("Drought Intensity (D2+)")
    ax.set_ylabel("Absolute Error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Random Forest Transfer Error vs Drought Intensity", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, "paper_rf_error_vs_drought.png")


print(f"Saved visualizations to: {VIZ_DIR}")
for path in sorted(VIZ_DIR.glob("*.png")):
    print(path.name)