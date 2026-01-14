import seaborn as sns

from .molgen_utils import (
    aggregate_molgen_fn,
    fp_name_to_fn,
    get_top_k_div_df,
    load_molgen_results,
)
from .molprop_utils import load_molprop_results
from .pandas_to_latex import PandasTableFormatter

CMAP_MODELS = {
    # Chem models — muted reds
    "ChemDFM-R": sns.color_palette("husl", n_colors=11)[0],
    "ChemDFM-v2.0": sns.color_palette("husl", n_colors=11)[1],
    "ether0": sns.color_palette("husl", n_colors=11)[2],
    # Reasoning models — cool blues / teals
    "Qwen3-A3B-Think.": sns.color_palette("husl", n_colors=11)[4],
    "gpt-oss": sns.color_palette("husl", n_colors=11)[5],
    "DeepSeek-R1-D.-Llama": sns.color_palette("husl", n_colors=11)[6],
    "DeepSeek-R1-D.-Qwen": sns.color_palette("husl", n_colors=11)[7],
    # Non-reasoning models — warm greens / golds
    "Llama-3.3-Instruct": sns.color_palette("husl", n_colors=11)[8],
    "gemma-3": sns.color_palette("husl", n_colors=11)[9],
}
MARKER_MODELS = {
    # Chem models
    "ChemDFM-R": "^",  # triangle up
    "ChemDFM-v2.0": "o",  # circle
    "ether0": "v",  # triangle down
    # Reasoning models
    "Qwen3-A3B-Think.": "s",  # square
    "gpt-oss": "D",  # diamond
    "DeepSeek-R1-D.-Llama": "P",  # filled plus
    "DeepSeek-R1-D.-Qwen": "X",  # filled x
    # Non-reasoning models
    "Llama-3.3-Instruct": "p",  # pentagon
    "gemma-3": "*",  # star
}


__all__ = [
    "PandasTableFormatter",
    "load_molgen_results",
    "aggregate_molgen_fn",
    "get_top_k_div_df",
    "CMAP_MODELS",
    "fp_name_to_fn",
    "load_molprop_results",
]
