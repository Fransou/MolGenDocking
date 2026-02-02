import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


class PandasTableFormatter:
    def __init__(
        self,
        n_decimals: int = 3,
        aggregation_methods: List[Any] = ["mean", "std"],
        main_subset: int = 0,
        total_col_name: str = "AVG.",
        hide_agg_labels: bool = True,
        already_rotated: bool = False,
        global_agg: bool = True,
        color_mapping: Union[Callable[[float], str], None] = None,
    ):
        """
        PandasTableFormatter is a class that formats a Pandas DataFrame into a LaTeX
        table with custom aggregation methods and styles.

        The 'main_subset' parameter is used to specify which aggregation methods should be used
        to compare the different rows in the table (often the mean). These aggregagtion results
        will be the one highlighted in the latex table.

        If main_subset is specified, a column ($NAME, agg$) will be created with the aggregation method

        :param color_mapping: Optional callable that takes a float value and returns a color string (e.g., '#FF5733').
                            This function will be applied to each cell to assign background colors based on values.
        """
        self.n_decimals = n_decimals
        self.aggregation_methods = aggregation_methods
        self.hide_agg_labels = hide_agg_labels
        self.already_rotated = already_rotated
        self.global_agg = global_agg
        self.color_mapping = color_mapping

        for agg in self.aggregation_methods:
            if not isinstance(agg, str) and not callable(agg):
                raise ValueError(
                    "Aggregation methods must be either a string or a callable function"
                )
            elif callable(agg):
                if not hasattr(agg, "__name__"):
                    raise ValueError(
                        "Aggregation function must have a __name__ attribute"
                    )
        assert 0 <= main_subset < len(aggregation_methods), (
            "main_subset must be an integer between 0 and the number of aggregation methods"
        )

        self.main_subset = main_subset
        if isinstance(aggregation_methods[main_subset], str):
            self.main_agg = aggregation_methods[main_subset]
        elif hasattr(aggregation_methods[main_subset], "__main__"):
            self.main_agg = aggregation_methods[main_subset].__name__
        self.total_col_name = total_col_name

    def _find_k_th_fn(
        self,
        s: np.ndarray | pd.Series,
        fn: Callable[[np.ndarray | pd.Series], float],
        k: int,
        props: str = "",
    ) -> List[str]:
        """
        Highlight the top k values in a dataframe using the prop string.

        :param s: The input array or series to be processed.
        :param fn: The function to be applied to the input.
        :param k: The number of top values to highlight.
        :param props: The properties to be applied to the highlighted values.

        :return: A list of strings with the highlighted values.
        """
        ps_list = [np.where(s == fn(s), True, False)]
        for i in range(1, k):
            previous_ps = np.concatenate(ps_list).reshape(i, -1).any(axis=0)
            ps_list.append(np.where(s == fn(s[~previous_ps]), True, False))
        return [props if p else "" for p in ps_list[-1]]

    def _apply_color_mapping(self, s: np.ndarray | pd.Series) -> List[str]:
        """
        Apply color mapping to a series of values based on the color_mapping function.

        :param s: The input array or series to be processed.
        :return: A list of CSS background-color properties for each value.
        """
        if self.color_mapping is None:
            return [""] * len(s)

        colors = []
        for val in s:
            if pd.isna(val):
                colors.append("")
            else:
                try:
                    color = self.color_mapping(float(val))
                    colors.append(f"background-color: {color};")
                except (ValueError, TypeError):
                    colors.append("")
        return colors

    def _aggregate_results_and_pivot(
        self,
        df_base: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
    ) -> pd.DataFrame:
        """
        Aggregates the given dataframe by computing the mean and standard deviation for a specified
        metric, organizing the results in a structured format.

        Example with no multicols:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
                    "category": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
                    "value": [0,1,2,-1,0,1,-1,1,3,-10,0,10],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg mean   avg std   A mean    A std    B mean      B std
            a     0.5        0.7        1         1        0          0.5
            b     0.5        0.7        1         2        0          10

        Example with multicols {"A": ["category1", "A"], "B": ["category1", "B"], "C": ["category2", "C"]}:
            df_base = pd.DataFrame(
                {
                    "name": ["a", "a", "a", "a", "b", "b", "b", "b"],
                    "category": ["A", "A", "B", "B", "A", "A", "B", "B"],
                    "value": [0,1,2,2,0,1,2,2],
                }
            )
            self._aggregate_results_with_std(df_base)

            >> Output:
                 avg    avg   A    A    B    B
                mean   std  mean  std  mean std
            a    0.5   0.5   1    1    2    0
            b    0.5   0.5   1    1    2    0


        :param df_base: The base dataframe containing the data to be aggregated.
        :param rows: The column(s) to be used as rows in the resulting dataframe.
        :param cols: The column(s) to be used as columns in the resulting dataframe.
        :param values: The column(s) to be used as values in the resulting dataframe.
        :param aggregation_methods: The aggregation functions to be applied to the values.

        :return:  A formatted dataframe containing aggregated mean and standard
        deviation values.
        """
        # Join the mean and std dataframes to a new one
        dataframes_to_concatenate = []
        df_glob = []
        for i_agg_meth, agg in enumerate(self.aggregation_methods):
            df_agg = df_base.pivot_table(
                index=rows, columns=cols, values=values, aggfunc=agg
            )
            df_agg.columns = pd.MultiIndex.from_arrays(
                [df_agg.columns.get_level_values(i) for i in range(len(cols))]
                + [
                    pd.Index(
                        [agg if isinstance(agg, str) else agg.__name__]
                        * df_agg.shape[1],
                        name="agg",
                    )
                ],
                names=cols + ["agg"] if isinstance(cols, list) else [cols, "agg"],
            )

            if i_agg_meth == self.main_subset and self.global_agg:
                for agg_b in self.aggregation_methods:
                    df_glob_agg = df_agg.agg(agg_b, axis=1)
                    df_glob_agg = df_glob_agg.to_frame(
                        name=agg_b if isinstance(agg_b, str) else agg_b.__name__
                    )
                    df_glob_agg.columns = pd.MultiIndex.from_arrays(
                        [
                            pd.Index(
                                [" " if i < len(cols) - 1 else self.total_col_name],
                                name=cols[i],
                            )
                            for i in range(len(cols))
                        ]
                        + [
                            pd.Index(
                                [agg_b if isinstance(agg_b, str) else agg_b.__name__],
                                name="agg",
                            )
                        ],
                        names=(
                            cols + ["agg"] if isinstance(cols, list) else [cols, "agg"]
                        ),
                    )
                    df_glob.append(df_glob_agg)
            dataframes_to_concatenate.append(df_agg)
        df_agg = pd.concat(dataframes_to_concatenate, axis=1)
        if self.global_agg:
            df_glob = pd.concat(df_glob, axis=1)
            df_agg = pd.concat([df_agg, df_glob], axis=1)
        df_agg.index.name = None
        try:
            # If one of the column's key contains a number, use the float of this number
            int_pattern = re.compile(r"(\d+)")
            idx_int: List[
                tuple[int, ...]
            ] = []  # idx of all levels that have an int, for each col
            for col_names in df_agg.columns:
                int_cols = []
                for i_elem, col_name in enumerate(col_names[:-1]):
                    if isinstance(col_name, str):
                        match = int_pattern.search(col_name)
                        if match:
                            int_cols.append(i_elem)
                idx_int.append(tuple(int_cols))
            if len(list(set(idx_int))) == 1:  # all columns have the same idx with int

                def key_fn(x: Any) -> Any:
                    key = []
                    for i, col_name in enumerate(x[:-1]):
                        if i in idx_int[0]:
                            match = int_pattern.search(col_name)
                            key.append(int(match.group(1)))  # type: ignore
                        else:
                            key.append(col_name)
                    return tuple(key)
            else:

                def key_fn(x: Any) -> Any:
                    return tuple(x[:-1])

            df_agg = df_agg.reindex(
                sorted(df_agg.columns, key=key_fn),
                axis=1,
            )
        except TypeError as e:
            print(e)
            warnings.warn(
                "Warning: Unable to sort columns. Ensure that the columns are of the same type."
            )
        return df_agg

    def style(
        self,
        df: pd.DataFrame,
        rows: str | List[str],
        cols: str | List[str],
        values: str,
        highlight_fn: Callable[[np.ndarray | pd.Series], float] = np.nanmax,
        props: List[str] = ["font-weight: bold;"],
        special_format_agg: Dict[str, Callable[[str], str]] = {
            "std": lambda x: "\\tiny $\\pm$" + x
        },
        remove_col_names: bool = False,
        row_order: Any | None = None,
    ) -> Styler:
        """
        Applies the highlight method to the given dataframe and returns a styled dataframe.
        If the Dataframe is already rotated, it will be melted first, where the columns denote
        the columns to keep, and value is the name given to the columns in the dataframe once melted.


        :param df: The dataframe to be styled.
        :return: A styled dataframe with highlighted values.
        """
        k = len(props)
        if self.already_rotated:
            df = df.melt(
                id_vars=rows,
                value_vars=cols,
                var_name=values,
                value_name="value",
            )
            cols = values
            values = "value"

        if isinstance(rows, str):
            rows = [rows]
        if isinstance(cols, str):
            cols = [cols]

        df_agg = self._aggregate_results_and_pivot(
            df,
            rows=rows,
            cols=cols,
            values=values,
        )

        # Trick as there is a data leakage (pandas issue)
        def wrap_special_format_agg(fn: Callable[[str], str]) -> Callable[[float], str]:
            def wrapped(x: float) -> str:
                x_str = str(np.round(x, self.n_decimals))
                out = fn(x_str)
                return out

            return wrapped

        formatter = {
            c: wrap_special_format_agg(special_format_agg[c[-1]])
            for c in df_agg.columns
            if c[-1] in special_format_agg
        }
        if remove_col_names and len(df_agg.columns.names) > 0:
            df_agg.columns = df_agg.columns.set_names(
                [
                    None,
                ]
                * len(df_agg.columns.names)
            )

        if row_order is not None:
            df_agg = df_agg.loc[row_order]
        style = df_agg.style.format(
            formatter,
            precision=self.n_decimals,
        )
        # Apply the highlight function to the specified columns
        for i in range(1, k + 1):
            style.apply(
                partial(self._find_k_th_fn, fn=highlight_fn, k=i, props=props[i - 1]),
                subset=([c for c in df_agg.columns if c[-1] == self.main_agg]),
            )

        # Apply color mapping to all cells if color_mapping is provided
        if self.color_mapping is not None:
            style.apply(self._apply_color_mapping)

        if self.hide_agg_labels:
            style = style.hide(axis="columns", level=df_agg.columns.nlevels - 1)
        return style

    def get_latex(
        self,
        style: Styler,
        cols_sep: Union[str, int, None] = 0,
        n_first_cols: int | None = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Returns the LaTeX representation of the styled dataframe.

        :param style: The styled dataframe to be converted to LaTeX.
        :param cols_sep: The column separator to use.
        :param kwargs: Additional arguments to be passed to the LaTeX conversion.

        :return: The LaTeX representation of the styled dataframe.
        """
        if "convert_css" in kwargs and not kwargs["convert_css"]:
            print(
                "Warning: 'convert_css' has to be set to True for most cases. Setting to True."
            )
            del kwargs["convert_css"]
        if cols_sep is not None:
            if "column_format" in kwargs:
                del kwargs["column_format"]
            column_format = ""
            if isinstance(cols_sep, int):
                for idx in style.data.index.names:
                    print(idx)
                    column_format += "c|"
                # cols reprents the level of the multindex we want to separate
                assert cols_sep <= style.data.columns.nlevels - 1
                prev_col_name = "************"
                for i_c, c in enumerate(style.data.columns):
                    if c[cols_sep] != prev_col_name:
                        column_format += "|c"
                        prev_col_name = c[cols_sep]
                    else:
                        column_format += "c"

            elif isinstance(cols_sep, str):
                column_format = cols_sep
        latex: str = style.to_latex(
            convert_css=True,
            column_format=column_format,
            **kwargs,
        )
        return latex

    def save_to_latex(
        self,
        style: Styler,
        filename: str = "table.tex",
        cols_sep: Union[str, int, None] = 0,
        **kwargs: Any,
    ) -> None:
        """
        Saves the styled dataframe to a LaTeX file.

        :param style: The styled dataframe to be saved.
        :param filename: The name of the LaTeX file.
        """
        latex = self.get_latex(style, cols_sep, **kwargs)
        with open(filename, "w") as f:
            f.write(latex)
