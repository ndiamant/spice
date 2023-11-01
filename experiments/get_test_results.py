import numpy as np
import wandb
import pandas as pd

from spice.datasets import get_baseline_size
from spice.utils import WANDB_PROJECT


def get_results(version: str) -> pd.DataFrame:
    api = wandb.Api()
    project_name = WANDB_PROJECT
    project = api.project(WANDB_PROJECT)
    runs = wandb.apis.public.Runs(
        project.client, project.entity, project_name,
        filters={"group": {"$in": [version]}},
    )
    df_rows = []
    for run in runs:
        df_row = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        df_row |= run.config
        df_rows.append(df_row)
    df = pd.DataFrame(df_rows)
    return df


def find_best_models(df: pd.DataFrame, metric_col: str, mode="min"):
    df = df.set_index(["dataset_name", "model"]).sort_index()
    df_out = df.groupby(level=[0, 1])[metric_col].agg(["mean", "sem"])
    df_out["rank"] = df_out.groupby(level=0)["mean"].rank(ascending=mode == "min")
    df_out["best"] = df_out["rank"] == 1
    df_out["second_best"] = df_out["rank"] == 2
    df_out.rename(columns={"mean": metric_col}, inplace=True)
    return df_out


def compute_mean_perf(df: pd.DataFrame, metric_col: str, mode="min"):
    df_out = df.reset_index().groupby("model", group_keys=False)[metric_col].agg(["mean", "sem"])
    df_out["rank"] = df_out["mean"].rank(ascending=mode == "min")
    df_out["best"] = df_out["rank"] == 1
    df_out["second_best"] = df_out["rank"] == 2
    df_out = df_out.reset_index()
    df_out["dataset_name"] = "mean"
    df_out.rename(columns={"mean": metric_col}, inplace=True)
    df_out.set_index(["dataset_name", "model"], inplace=True)
    df_out = pd.concat([df, df_out])
    return df_out


def to_line(entries: list[str]) -> str:
    return " & ".join(entries) + " \\\\"


def build_latex_table(
    df: pd.DataFrame, metric: str, models: dict[str, str], datasets: dict[str, str], rounding=2, mode: str = "min",
    include_mean: bool = True,
):
    """
    \begin{center}
    \begin{tabular}{lccccc}
    \textsc{MODEL}  & \textsc{bike} & \textsc{bio} \\
    \midrule
    CQR         & \\
    CHR             & \\
    PCP             & \\
    Histogram & \\
    \methodname ($n=1$) \\
    \methodname ($n=2$) \\
    \methodname-HPD ($n=1$) \\
    \methodname-HPD ($n=2$) \\

    \end{tabular}
    \end{center}
    """
    df = find_best_models(df[df["model"].isin(models) & df["dataset_name"].isin(datasets)], metric, mode)
    if include_mean:
        df = compute_mean_perf(df, metric, mode)
        datasets["mean"] = "mean"

    # header
    print("\\begin{{tabular}}{{l{}}}".format("c" * len(datasets)))
    line = ["\\textsc{MODEL}"]
    line += ["\\textsc{{{}}}".format(dataset) for dataset in datasets.values()]
    print(to_line(line))
    print("\\midrule")

    # metric results
    for model, display_name in models.items():
        line = [display_name]
        for dataset in datasets:
            val = round(df.loc[(dataset, model), metric], rounding)
            sem = df.loc[(dataset, model), "sem"]
            disp_val = f"{val:.2f} \\pm {sem:.2f}"
            best = df.loc[(dataset, model), "best"]
            second_best = df.loc[(dataset, model), "second_best"]
            if best:
                disp_val = f"\\mathbf{{{disp_val}}}"
            if second_best:
                disp_val = f"\\underline{{{disp_val}}}"

            line.append(f"${disp_val}$")
        print(to_line(line))

    print("\\end{tabular}")


if __name__ == "__main__":
    # download results from wandb
    dfs = [
        "cqr", "pcp", "chr", "conditional_hist",
        "conditional_hist_HPD",
        "spice_n2", "spice_n2_HPD",
        "spice_n1", "spice_n1_HPD",
    ]
    dfs = {
        name: get_results("10-03_eval_v0")
        for name in dfs
    }
    # update HPD results
    for name, df in dfs.items():
        if "_HPD" not in name: continue
        df["model"] = df["model"] + "_HPD"
        for col in df:
            if "hpd_" not in col: continue
            no_hpd_name = col.replace("hpd_", "")
            df[no_hpd_name] = df[col]
    df = pd.concat(list(dfs.values()))
    # get normalized size
    df["normalized_size"] = np.nan
    for name in df["dataset_name"].unique():
        idx = df["dataset_name"] == name
        df.loc[idx, "normalized_size"] = (
            df.loc[idx, "test/size_at_0.1"]
            / get_baseline_size(name, 0.1)
        )
    # normalized size results
    datasets = {
        "bike": "bike", "bio": "bio", "blog_data_og": "blog", "meps_19_og": "meps19", "meps_20_og": "meps20",
        "meps_21_og": "meps21", "star": "star", "temperature": "temp.",
    }
    models = {
        "cqr": "CQR", "chr": "CHR", "pcp": "PCP", "conditional_hist": "Hist.",
        "spice_n1": "\\methodnameNegDensity $(n=1$)", "spice_n1_HPD": "\\methodnameHPD $(n=1$)",
        "spice_n2": "\\methodnameNegDensity $(n=2$)", "spice_n2_HPD": "\\methodnameHPD $(n=2$)",
    }
    build_latex_table(
        df, "normalized_size",
        models=models,
        datasets=datasets,
    )
    # conditional coverage results
    build_latex_table(
        df, "test/y_stratified_coverage_at_0.1",
        models=models,
        datasets=datasets,
        mode="max",
    )
