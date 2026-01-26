import seaborn as sns

from .molgen_utils import (
    aggregate_molgen_fn,
    fp_name_to_fn,
    get_top_k_div_df,
    load_molgen_results,
)
from .molprop_utils import load_molprop_results
from .pandas_to_latex import PandasTableFormatter
from .synthesis_utils import load_synth_results

N_COLORS = 13
CMAP_MODELS = {
    # Chem models — muted reds
    "ChemDFM-R": sns.color_palette("husl", n_colors=N_COLORS)[0],
    "ChemDFM-v2.0": sns.color_palette("husl", n_colors=N_COLORS)[1],
    "ether0": sns.color_palette("husl", n_colors=N_COLORS)[2],
    "RL-Mistral": sns.color_palette("husl", n_colors=N_COLORS)[3],
}

CMAP_MODELS.update(
    {
        # Reasoning models — cool blues / teals
        "MiniMax-M2": sns.color_palette("husl", n_colors=N_COLORS)[
            len(CMAP_MODELS) + 1
        ],
        "Qwen3-Think.": sns.color_palette("husl", n_colors=N_COLORS)[
            len(CMAP_MODELS) + 2
        ],
        "Qwen3-Next-Think.": sns.color_palette("husl", n_colors=N_COLORS)[
            len(CMAP_MODELS) + 3
        ],
        "gpt-oss": sns.color_palette("husl", n_colors=N_COLORS)[len(CMAP_MODELS) + 4],
        "R1-Llama": sns.color_palette("husl", n_colors=N_COLORS)[len(CMAP_MODELS) + 5],
        "R1-Qwen": sns.color_palette("husl", n_colors=N_COLORS)[len(CMAP_MODELS) + 6],
    }
)

CMAP_MODELS.update(
    {
        # Non-reasoning models — warm greens / golds
        "Llama-3.3": sns.color_palette("husl", n_colors=N_COLORS)[len(CMAP_MODELS) + 1],
        "gemma-3": sns.color_palette("husl", n_colors=N_COLORS)[len(CMAP_MODELS) + 2],
    }
)
MARKER_MODELS = {
    # Chem models
    "ChemDFM-R": "^",  # triangle up
    "ChemDFM-v2.0": "o",  # circle
    "ether0": "v",  # triangle down
    "RL-Mistral": "X",  # filled x
    # Reasoning models
    "MiniMax-M2": "h",  # hexagon
    "Qwen3-Think.": "s",  # square
    "Qwen3-Next-Think.": "^",  # triangle up
    "gpt-oss": "D",  # diamond
    "R1-Llama": "P",  # filled plus
    "R1-Qwen": "X",  # filled x
    # Non-reasoning models
    "Llama-3.3": "p",  # pentagon
    "gemma-3": "*",  # star
}

HIGHLIGHT_MODELS = ["ChemDFM-R", "MiniMax-M2", "RL-Mistral"]

__all__ = [
    "PandasTableFormatter",
    "load_molgen_results",
    "aggregate_molgen_fn",
    "get_top_k_div_df",
    "CMAP_MODELS",
    "fp_name_to_fn",
    "load_molprop_results",
    "load_synth_results",
]
