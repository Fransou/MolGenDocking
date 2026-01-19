import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from notebooks.utils import *
from notebooks.utils import (
    CMAP_MODELS,
    HIGHLIGHT_MODELS,
    MARKER_MODELS,
    get_top_k_div_df,
    load_molgen_results,
)

MOLSTRAL_PATH = Path("MolGenOutput/test_ood")
FIG_PATH = "/home/philippe/-Philippe-MolGenDocking/Figures/Results/MolGen"
os.makedirs(FIG_PATH, exist_ok=True)

files = [
    f
    for d in MOLSTRAL_PATH.iterdir()
    for f in d.iterdir()
    if "error" not in str(f) and str(f).endswith("scored.jsonl")
]
files = sorted(files)
print("Total files:", len(files))
df = load_molgen_results(files[:])

sub_sample_prompts = df[df.model == "MiniMax-M2_"].prompt_id.unique()[:]
df = df[df.prompt_id.isin(sub_sample_prompts)]

sub_sample_prompts = df[df.model == "Qwen3-Next-80B-A3B-Thinking_"].prompt_id.unique()[
    :
]
df = df[df.prompt_id.isin(sub_sample_prompts)]


fig_name = FIG_PATH + "/diversity_rewards/{}"
os.makedirs(FIG_PATH + "/diversity_rewards", exist_ok=True)


def plot_div_topk(
    df: pd.DataFrame,
    fp_name: str,
    legend: bool = False,
    cols_vals: list[float] = [10, 30],
    x_vals: list[float] | None = None,
    row: str = "n_rollout",
    col: str = "k",
    x: str = "sim",
    column_name: str = "k",
    xlabel: str = "Similarity threshold between candidate clusters",
    xlabel_x: float = 0.3,
    legend_bbox: tuple[float, float] = (0.3, -0.2),
    legend_ncols: int = 4,
    **kwargs: Any,
) -> sns.FacetGrid:
    def draw(data: pd.DataFrame, **kwargs: Any) -> None:
        # Separate highlighted and non-highlighted models
        highlighted_data = data[data["Model"].isin(HIGHLIGHT_MODELS)]
        non_highlighted_data = data[~data["Model"].isin(HIGHLIGHT_MODELS)]

        # Draw non-highlighted models with normal styling
        if len(non_highlighted_data) > 0:
            sns.lineplot(
                non_highlighted_data,
                x=x,
                y="value",
                hue="Model",
                sizes=1,
                alpha=0.7,
                palette=CMAP_MODELS,
                hue_order=CMAP_MODELS.keys(),
                dashes=False,
                legend=legend,
                linewidth=1.0,
                **kwargs,
            )
            sns.lineplot(
                non_highlighted_data.iloc[2::3],
                x=x,
                y="value",
                hue="Model",
                style="Model",
                sizes=1,
                alpha=0.7,
                palette=CMAP_MODELS,
                hue_order=CMAP_MODELS.keys(),
                markers=MARKER_MODELS,
                dashes=False,
                legend=legend,
                markersize=4.0,
                linewidth=0.0,
                markeredgewidth=0.0,
                **kwargs,
            )

        # Draw highlighted models with enhanced styling (thicker lines, larger markers)
        if len(highlighted_data) > 0:
            sns.lineplot(
                highlighted_data,
                x=x,
                y="value",
                hue="Model",
                sizes=1,
                alpha=1,
                palette=CMAP_MODELS,
                hue_order=HIGHLIGHT_MODELS,
                dashes=False,
                legend=legend,
                linewidth=1.4,
                **kwargs,
            )
            sns.lineplot(
                highlighted_data.iloc[2::3],
                x=x,
                y="value",
                hue="Model",
                style="Model",
                sizes=1,
                alpha=1,
                palette=CMAP_MODELS,
                hue_order=HIGHLIGHT_MODELS,
                markers=MARKER_MODELS,
                dashes=False,
                legend=legend,
                markersize=4.8,
                linewidth=0.0,
                markeredgewidth=0.0,
                **kwargs,
            )

    div_clus_df = get_top_k_div_df(
        df[df.prompt_id.isin(sub_sample_prompts[:5])], fp_name=fp_name, **kwargs
    )
    if cols_vals is not None:
        div_clus_df = div_clus_df[div_clus_df[col].isin(cols_vals)]
    if x_vals is not None:
        div_clus_df = div_clus_df[div_clus_df[x].isin(x_vals)]
    g = sns.FacetGrid(
        div_clus_df,
        row=row,
        col=col,
        margin_titles=True,
        height=1.5,
        aspect=1.7,
    )
    g.map_dataframe(draw)
    # Add legend to the top right
    g.set_axis_labels("", "")
    # set x axis log
    g.set(xscale="log")

    g.set_titles(
        row_template="$n_r$={row_name}", col_template=column_name + "={col_name}"
    )
    g.fig.supxlabel(xlabel, y=0.0, x=xlabel_x)
    g.fig.supylabel("Diversity-Aware Top-k Score", x=0.04)
    if legend:
        g.add_legend(
            title="Model",
            loc="lower center",
            bbox_to_anchor=legend_bbox,
            ncols=legend_ncols,
            fontsize=8,
            title_fontsize=10,
        )
    # Add grid
    for ax in g.axes.flatten():
        ax.grid(True, which="both", linestyle="--", linewidth=0.05, alpha=0.3)
    return g


if __name__ == "__main__":
    MAIN_FP = "ecfp6-2048"
    common_kwargs = {
        "df": df,
        "column_name": "k",
        "xlabel": "Similarity threshold between candidate clusters",
        "sim_values": np.logspace(-1.2, -0.01, 18).tolist(),
    }

    g = plot_div_topk(
        fp_name=MAIN_FP,
        cols_vals=[
            5,
            10,
            20,
        ],
        legend=True,
        xlabel_x=0.3,
        legend_bbox=(0.3, -0.2),
        **common_kwargs,
    )
    plt.savefig(f"{fig_name.format(MAIN_FP) + '-main'}.pdf", bbox_inches="tight")
    plt.show()

    for fp_name in [
        "ecfp4-2048",
        "ecfp6-2048",
        "maccs",
        "rdkit",
        "Avalon",
        "Gobbi2d",
    ]:
        g = plot_div_topk(
            fp_name=fp_name,
            cols_vals=[
                5,
                10,
            ],
            legend=fp_name == "Avalon",
            xlabel_x=0.5,
            legend_bbox=(0.9, 0.5),
            legend_ncols=1,
            **common_kwargs,
        )
        g.fig.suptitle(fp_name, y=1.05)
        plt.savefig(f"{fig_name.format(fp_name)}.pdf", bbox_inches="tight")
        plt.show()

    g = plot_div_topk(
        df=df,
        fp_name=MAIN_FP,
        col="sim",
        x="k",
        k_max=50,
        cols_vals=[0.3, 0.5, 0.7, 0.8],
        sim_values=[0.3, 0.5, 0.7, 0.8],
        x_vals=[1.0] + [float(i) for i in range(5, 50, 5)],
        legend=True,
        column_name="Sim. thresh.",
        xlabel="k",
        xlabel_x=0.3,
        legend_bbox=(0.3, -0.2),
        legend_ncols=4,
    )

    plt.savefig(f"{fig_name.format(MAIN_FP) + '_kx-main'}.pdf", bbox_inches="tight")
    plt.show()
