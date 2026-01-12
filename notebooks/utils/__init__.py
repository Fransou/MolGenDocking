from .molgen_utils import aggregate_molgen_fn, get_top_k_div_df, load_molgen_results
from .pandas_to_latex import PandasTableFormatter

CMAP_MODELS = {
    "ChemDFM-R": "orange",
    "ChemDFM-v2.0": "goldenrod",
    "ether0": "chocolate",
    "DeepSeek-R1-D.-Llama": "darkorchid",
    "DeepSeek-R1-D.-Qwen": "orchid",
    "Llama-3.3-Instruct": "darkslategray",
    "Qwen3-A3B-Think.": "teal",
    "gemma-3": "crimson",
}

__all__ = [
    "PandasTableFormatter",
    "load_molgen_results",
    "aggregate_molgen_fn",
    "get_top_k_div_df",
    "CMAP_MODELS",
]
