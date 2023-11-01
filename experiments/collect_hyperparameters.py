import wandb
import pandas as pd

from spice.utils import WANDB_PROJECT, USERNAME


def get_results(version: str) -> pd.DataFrame:
    api = wandb.Api()
    project = api.project(WANDB_PROJECT)
    runs = wandb.apis.public.Runs(
        project.client, project.entity, WANDB_PROJECT,
        filters={"group": {"$in": [version]}},
    )
    df_rows = []
    for run in runs:
        df_row = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        df_row |= run.config
        df_rows.append(df_row)
    df = pd.DataFrame(df_rows)
    return df


def get_best_hyperparams(df: pd.DataFrame, extra_hps: list[str] = None, alpha=0.1, tol=2.5e-2) -> pd.DataFrame:
    extra_hps = extra_hps or []
    hps = ["lr", "wd"] + extra_hps
    df_idx = df.set_index(["dataset_name"] + hps).sort_index()
    df_idx["val/mean_size"] = df_idx.groupby(level=list(range(len(hps) + 1)))[f"val/size_at_{alpha}"].mean()
    df_idx["val/mean_coverage"] = df_idx.groupby(level=list(range(len(hps) + 1)))[f"val/coverage_at_{alpha}"].mean()
    print(len(df_idx))
    df_idx = df_idx.loc[(df_idx["val/mean_coverage"] - (1 - alpha)).abs() < tol]
    print(len(df_idx))
    best_size = df_idx.groupby(level=0)["val/mean_size"].min()
    df_idx = df_idx.reset_index().set_index("dataset_name")
    df_idx["val/best_size"] = best_size
    best_hparams = df_idx[df_idx["val/mean_size"] == df_idx["val/best_size"]]
    best_hparams = best_hparams.groupby(level=0).first()  # when ties, take arbitrary
    return best_hparams


def save_hyperparams(df: pd.DataFrame, name: str, hps: list[str]):
    df = df[hps]
    df = df.reset_index()
    df.to_csv(f"/home/{USERNAME}/repos/splintile/experiments/hyperparameters/{name}.csv", index=False)


if __name__ == "__main__":
    hist_df = get_results("09-22-2023_cond_hist_v1")
    hist_df["model"] = "histogram"
    cqr_df = get_results("09-21-2023_cqr_v0")
    cqr_df["model"] = "CQR"
    pcp_df = get_results("09-21-2023_pcp_v1")
    pcp_df["model"] = "PCP"
    chr_df = get_results("09-21-2023_chr_v0")
    chr_df["model"] = "CHR"
    spice_n2_df = get_results("09-22-2023_spiceN2_v0")
    spice_n2_df["model"] = "spice_n1"
    spice_n1_df = get_results("09-22-2023_spice_n1_v1")
    spice_n1_df["model"] = "spice_n1"

    alpha = 0.1
    shared_hps = ["lr", "wd"]
    metric_cols = [f"val/mean_size", f"val/mean_coverage"]

    # hist.
    extras = ["n_bins", "smart_bin_positions"]
    hist_hps = get_best_hyperparams(hist_df, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(hist_hps, "cond_hist", shared_hps + extras)
    hist_hpd = hist_df.copy()
    hist_hpd[[f"val/size_at_{alpha}", f"val/coverage_at_{alpha}"]] = hist_hpd[
        [f"val/hpd_size_at_{alpha}", f"val/hpd_coverage_at_{alpha}"]]
    hist_hps_hpd = get_best_hyperparams(hist_hpd, extras, alpha=alpha, tol=0.1)[shared_hps + extras + metric_cols]
    save_hyperparams(hist_hps_hpd, "cond_hist_hpd", shared_hps + extras)

    # chr
    extras = ["n_bins"]
    chr_hps = get_best_hyperparams(
        chr_df, extras, alpha=alpha,
    )[shared_hps + extras + metric_cols]
    save_hyperparams(chr_hps, "chr", shared_hps + extras)

    # cqr
    extras = ["qr_interval"]
    cqr_hps = get_best_hyperparams(cqr_df, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(cqr_hps, "cqr", shared_hps + extras)

    # pcp
    extras = ["n_mixture"]
    pcp_hps = get_best_hyperparams(pcp_df, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(pcp_hps, "pcp", shared_hps + extras)

    extras = ["n_knots", "learn_bin_widths", "min_f_bar_val"]
    spice_n2_hps = get_best_hyperparams(spice_n2_df, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(spice_n2_hps, "spice_n2", shared_hps + extras)

    # spice n = 2
    spice_n2_hpd = spice_n2_df.copy()
    spice_n2_hpd[[f"val/size_at_{alpha}", f"val/coverage_at_{alpha}"]] = spice_n2_hpd[
        [f"val/hpd_approx_size_at_{alpha}", f"val/hpd_coverage_at_{alpha}"]]
    spice_n2_hps_hpd = get_best_hyperparams(spice_n2_hpd, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(spice_n2_hps_hpd, "spice_n2_hpd", shared_hps + extras)

    extras = ["n_knots", "learn_bin_widths", "min_likelihood"]
    spice_n1_hps = get_best_hyperparams(spice_n1_df, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(spice_n1_hps, "spice_n1", shared_hps + extras)

    # spice n = 1
    spice_n1_hpd = spice_n1_df.copy()
    spice_n1_hpd[[f"val/size_at_{alpha}", f"val/coverage_at_{alpha}"]] = spice_n1_hpd[
        [f"val/hpd_size_at_{alpha}", f"val/hpd_coverage_at_{alpha}"]]
    spice_n1_hps_hpd = get_best_hyperparams(spice_n1_hpd, extras, alpha=alpha)[shared_hps + extras + metric_cols]
    save_hyperparams(spice_n1_hps_hpd, "spice_n1_hpd", shared_hps + extras)
