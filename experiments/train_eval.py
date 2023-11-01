import argparse
import os

import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from spice.conditional_histogram import ConditionalHist
from spice.chr import CHR
from spice.datasets import RegressionData
from spice.cqr import CQR
from spice.pcp import PCP
from spice.utils import timestamp, rename_metrics, WANDB_PROJECT
from spice.spice_n2 import SPICEn2, smart_bin_init
from spice.spice_n1 import SPICEn1
from spice.spice_n1 import smart_bin_init as spice_n1_smart_bin_init


def setup_trainer_and_data(
    name: str, wandb_log_dir: str,
    epochs: int, version: str, checkpoint_folder: str,
    dataset_name: str, seed: int,
    y_scaling: str = "min_max", discretize_n_bins: int = None,
    smart_discretize: bool = True,
) -> tuple[Trainer, WandbLogger, ModelCheckpoint, RegressionData]:
    data = RegressionData(
        dataset_name, train_seed=seed, y_scaling=y_scaling, discretize_n_bins=discretize_n_bins,
        smart_discretize=smart_discretize,
    )
    logger = WandbLogger(
        project=WANDB_PROJECT, save_dir=wandb_log_dir,
        name=name, group=version,
        version=f"{version}_{name}",
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_folder, name)
    )
    max_steps_per_epoch = 100
    max_val_steps = 10
    train_batches = data.train_batches(max_steps_per_epoch)
    trainer = Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=epochs // 4, mode="min"),
            LearningRateMonitor(),
            checkpoint,
        ],
        accelerator="gpu", max_steps=epochs * max_steps_per_epoch,
        check_val_every_n_epoch=1,
        limit_train_batches=train_batches,
        limit_val_batches=data.val_batches(max_val_steps),
        enable_progress_bar=False,
        gradient_clip_val=5,
        log_every_n_steps=train_batches,
    )
    return trainer, logger, checkpoint, data


def run_conditional_histogram(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    n_bins: int,
    seed: int,
    alphas: list[float],
    smart_bin_positions: bool,
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    # set up data
    ts = timestamp()
    name = f"conditional_hist_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
        discretize_n_bins=n_bins, smart_discretize=smart_bin_positions,
    )
    seed_everything(seed)
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "conditional_hist",
        "n_bins": n_bins, "smart_bin_positions": smart_bin_positions,
        "seed": seed,
    })
    # set up model
    x_train, y_train = data.train_dset.tensors
    model = ConditionalHist(
        input_dim=x_train.shape[1], hidden_dim=hidden, bins=data.bins,
        lr=lr, wd=wd, max_iter=trainer.max_steps, y_min=0.0,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path)
    model: ConditionalHist = model.eval()
    # run conformal
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    thresholds = []
    hpd_thresholds = []
    for alpha in alphas:
        threshold = model.find_prob_threshold(x_cal, y_cal, alpha)
        thresholds.append(threshold)
        metrics = model.get_metrics(x_cal_val, y_cal_val, threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
        # hpd
        hpd_threshold = model.get_hpd_threshold(x_cal, y_cal, alpha)
        hpd_thresholds.append(hpd_threshold)
        metrics = model.get_hpd_metrics(x_cal_val, y_cal_val, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
    # testing
    if not run_test:
        wandb.finish()
        return model, data, thresholds
    x_test, y_test = data.test_dset.tensors
    for alpha, threshold, hpd_threshold in zip(alphas, thresholds, hpd_thresholds):
        thresholds.append(threshold)
        metrics = model.get_metrics(x_test, y_test, threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))
        # hpd
        hpd_thresholds.append(hpd_threshold)
        metrics = model.get_hpd_metrics(x_test, y_test, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def run_cqr(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    seed: int,
    alphas: list[float],
    qr_interval: float,
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    ts = timestamp()
    name = f"cqr_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
        y_scaling="std",
    )
    seed_everything(seed)
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "cqr",
        "qr_interval": qr_interval, "seed": seed,
    })
    # set up model
    x_dim = data.train_dset.tensors[0].shape[1]
    low_quantile = round((1 - qr_interval) / 2, 3)
    high_quantile = 1 - low_quantile
    model = CQR(
        input_dim=x_dim, hidden_dim=hidden,
        lr=lr, wd=wd, max_iter=trainer.max_steps,
        low_quantile=low_quantile, high_quantile=high_quantile,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path)
    model: CQR = model.eval()
    # run conformal
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    q_hats = []
    for alpha in alphas:
        q_hat = model.get_q_hat(x_cal, y_cal, alpha)
        metrics = model.get_metrics(
            x_cal_val, y_cal_val, q_hat,
        )
        metrics["size"] /= data.y_min_max_scaler.data_range_.item()
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
        q_hats.append(q_hat)
    # testing
    if not run_test:
        wandb.finish()
        return model, data, q_hats
    x_test, y_test = data.test_dset.tensors
    for alpha, q_hat in zip(alphas, q_hats):
        metrics = model.get_metrics(
            x_test, y_test, q_hat,
        )
        metrics["size"] /= data.y_min_max_scaler.data_range_.item()
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def run_pcp(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    seed: int,
    alphas: list[float],
    n_mixture: int,
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    ts = timestamp()
    name = f"pcp_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
        y_scaling="std",
    )
    seed_everything(seed)
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "pcp", "seed": seed,
    })
    # set up model
    x_dim = data.train_dset.tensors[0].shape[1]
    model = PCP(
        input_dim=x_dim, hidden_dim=hidden, max_iter=trainer.max_steps,
        lr=lr, wd=wd, n_mixture=n_mixture,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path)
    model: PCP = model.eval()
    # run conformal
    interval_workers = 8
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    q_hats = []
    for alpha in alphas:
        q_hat = model.get_q_hat(x_cal, y_cal, alpha)
        q_hats.append(q_hat)
        metrics = model.get_metrics(
            x_cal_val, y_cal_val, q_hat, interval_workers=interval_workers,
        )
        metrics["size"] /= data.y_min_max_scaler.data_range_.item()
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
    # testing
    if not run_test:
        wandb.finish()
        return model, data, q_hats
    x_test, y_test = data.test_dset.tensors
    for alpha, q_hat in zip(alphas, q_hats):
        metrics = model.get_metrics(
            x_test, y_test, q_hat, interval_workers=interval_workers,
        )
        metrics["size"] /= data.y_min_max_scaler.data_range_.item()
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def run_chr(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    n_bins: int,
    seed: int,
    alphas: list[float],
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    # set up data
    ts = timestamp()
    name = f"chr_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
    )
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "chr", "seed": seed,
    })
    # set up model
    x_train, y_train = data.train_dset.tensors
    model = CHR(
        input_dim=x_train.shape[1], hidden_dim=hidden, n_bins=n_bins,
        lr=lr, wd=wd, max_iter=trainer.max_steps,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path)
    model: CHR = model.eval()
    # run conformal
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    # n = len
    # x_cal, y_cal = x_cal[:n], y_cal[:n]
    # x_cal_val, y_cal_val = x_cal_val[:n], y_cal_val[:n]
    chrs = []
    for alpha in alphas:
        chr_ = model.calibrate(x_cal, y_cal, alpha)
        chrs.append(chr_)
        metrics = model.get_metrics(x_test=x_cal_val, y_test=y_cal_val, chr=chr_)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
    # testing
    if not run_test:
        wandb.finish()
        return model, data, chrs
    x_test, y_test = data.test_dset.tensors
    for alpha, chr_ in zip(alphas, chrs):
        metrics = model.get_metrics(x_test, y_test, chr_)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def run_spice_n2(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    n_knots: int,
    seed: int,
    alphas: list[float],
    learn_bin_widths: bool,
    min_f_bar: float,
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    ts = timestamp()
    name = f"spice_n2_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
    )
    seed_everything(seed)
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "spice_n2",
        "learn_bin_width": learn_bin_widths, "seed": seed,
    })
    # set up model
    x_train, y_train = data.train_dset.tensors
    y_train = y_train.squeeze()
    w, h = smart_bin_init(y_train, n_knots)
    model = SPICEn2(
        input_dim=x_train.shape[1], hidden_dim=hidden, n_knots=n_knots,
        lr=lr, wd=wd, max_iter=trainer.max_steps,
        learn_bin_widths=learn_bin_widths,
        smart_bin_init_h=h,
        smart_bin_init_w=w,
        min_f_bar_val=min_f_bar,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path, y_train=y_train.squeeze())
    model: SPICEn2 = model.eval().cuda()
    # run conformal
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    thresholds = []
    hpd_thresholds = []
    for alpha in alphas:
        # bayesian
        threshold = model.get_threshold(x_cal, y_cal, alpha)
        thresholds.append(threshold)
        metrics = model.get_metrics(x_cal_val, y_cal_val, threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
        # hpd
        hpd_threshold = model.get_hpd_threshold(x_cal, y_cal, alpha)
        hpd_thresholds.append(hpd_threshold)
        metrics = model.get_hpd_metrics(x_cal_val, y_cal_val, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
    # testing
    if not run_test:
        wandb.finish()
        return model, data, thresholds
    x_test, y_test = data.test_dset.tensors
    for alpha, threshold, hpd_threshold in zip(alphas, thresholds, hpd_thresholds):
        # bayesian
        metrics = model.get_metrics(x_test, y_test, threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))
        # hpd
        metrics = model.get_hpd_metrics(x_test, y_test, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def run_spice_n1(
    dataset_name: str,
    lr: float, wd: float, epochs: int,
    hidden: int,
    n_knots: int,
    seed: int,
    alphas: list[float],
    learn_bin_widths: bool,
    min_likelihood: float,
    # saving settings
    checkpoint_folder: str, version: str, wandb_log_dir: str,
    #
    run_test: bool = False,
):
    ts = timestamp()
    name = f"spice_n1_version-{version}_{ts}"
    trainer, logger, checkpoint, data = setup_trainer_and_data(
        name=name, wandb_log_dir=wandb_log_dir, epochs=epochs, version=version,
        dataset_name=dataset_name, seed=seed, checkpoint_folder=checkpoint_folder,
    )
    seed_everything(seed)
    wandb.config.update({
        "dataset_name": dataset_name, "alphas": alphas, "model": "spice_n1", "seed": seed,
    })
    # set up model
    x_train, y_train = data.train_dset.tensors
    y_train = y_train.squeeze()
    w, h = spice_n1_smart_bin_init(y_train, n_knots)
    model = SPICEn1(
        input_dim=x_train.shape[1], hidden_dim=hidden, n_knots=n_knots,
        lr=lr, wd=wd, max_iter=trainer.max_steps,
        learn_bin_widths=learn_bin_widths,
        bin_height_init=h,
        bin_width_init=w,
        min_likelihood=min_likelihood,
    )
    # fit model
    trainer.fit(model, datamodule=data)
    model = model.load_from_checkpoint(checkpoint.best_model_path, y_train=y_train.squeeze())
    model: SPICEn1 = model.eval().cuda()
    # run conformal
    x_cal, y_cal = data.cal_dset.tensors
    x_cal_val, y_cal_val = data.cal_val_dset.tensors
    thresholds = []
    hpd_thresholds = []
    for alpha in alphas:
        # bayesian
        threshold = model.get_threshold(x_cal, y_cal, alpha)
        thresholds.append(threshold)
        metrics = model.get_metrics(x_cal_val, y_cal_val, threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
        # hpd
        hpd_threshold = model.get_hpd_threshold(x_cal, y_cal, alpha)
        hpd_thresholds.append(hpd_threshold)
        metrics = model.get_hpd_metrics(x_cal_val, y_cal_val, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "val", alpha))
    # testing
    if not run_test:
        wandb.finish()
        return model, data, thresholds
    x_test, y_test = data.test_dset.tensors
    for alpha, threshold, hpd_threshold in zip(alphas, thresholds, hpd_thresholds):
        # bayesian
        metrics = model.get_metrics(x_test, y_test, threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))
        # hpd
        metrics = model.get_hpd_metrics(x_test, y_test, hpd_threshold)
        logger.log_metrics(rename_metrics(metrics, "test", alpha))


def bool_type(arg: str) -> bool:
    if arg.lower() == "false":
        return False
    if arg.lower() == "true":
        return True
    raise ValueError(f"Unrecognized argument {arg}. Should be True or False.")


def add_spice_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument('--n_knots', type=int, required=True, help='Number of knots')
    parser.add_argument(
        '--smart_bin_positions', type=bool_type, required=True,
        help='Whether to set the bin x positions in a smart way.',
    )
    add_shared_parser_args(parser)


def add_conditional_hist_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument('--n_bins', type=int, required=True, help='Number of bins')
    parser.add_argument(
        '--smart_bin_positions', type=bool_type, required=True,
        help='Whether to set the bin x positions in a smart way.',
    )
    add_shared_parser_args(parser)


def add_chr_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument('--n_bins', type=int, required=True, help='Number of bins')
    add_shared_parser_args(parser)


def add_cqr_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument(
        '--qr_interval', type=float, required=True,
        help='Range of quantile regression (e.g. 0.8 -> low quantile, high quantile = 0.05, 0.95',
    )
    add_shared_parser_args(parser)


def add_pcp_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument('--n_mixture', type=int, required=True, help='Number mixture components for normal mixture')
    add_shared_parser_args(parser)


def add_spice_n2_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument("--learn_bin_widths", type=bool_type, required=True)
    parser.add_argument("--min_f_bar", type=float, required=True)
    parser.add_argument("--n_knots", type=int, required=True)
    add_shared_parser_args(parser)


def add_spice_n1_parser(parser: argparse.ArgumentParser):
    # Modeling arguments
    parser.add_argument("--learn_bin_widths", type=bool_type, required=True)
    parser.add_argument("--min_likelihood", type=float, required=True)
    parser.add_argument("--n_knots", type=int, required=True)
    add_shared_parser_args(parser)


def add_shared_parser_args(parser: argparse.ArgumentParser):
    # model settings
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--wd', type=float, required=True, help='Weight decay')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--hidden', type=int, required=True, help='Number of hidden units')
    parser.add_argument('--seed', type=int, required=True, help='Seed for random number generator')
    parser.add_argument(
        '--alphas', type=float, required=True, help='Conformal rate of mis-specified prediction sets',
        nargs='+',
    )
    # Saving settings arguments
    parser.add_argument('--checkpoint_folder', type=str, required=True, help='Folder to save checkpoints')
    parser.add_argument('--version', type=str, required=True, help='Version of the model')
    parser.add_argument('--wandb_log_dir', type=str, required=True, help='Log directory for Weights and Biases')
    # Additional settings
    parser.add_argument('--run_test', type=bool_type, default=False, help='Whether to run test or not')


if __name__ == "__main__":
    # set up parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Which model to run.", dest="mode", description="which model to run")
    spice_parser = subparsers.add_parser("spice")
    add_spice_parser(spice_parser)
    hist_parser = subparsers.add_parser("conditional_hist")
    add_conditional_hist_parser(hist_parser)
    cqr_parser = subparsers.add_parser("cqr")
    add_cqr_parser(cqr_parser)
    pcp_parser = subparsers.add_parser("pcp")
    add_pcp_parser(pcp_parser)
    chr_parser = subparsers.add_parser("chr")
    add_chr_parser(chr_parser)
    spice_n2_parser = subparsers.add_parser("spice_n2")
    add_spice_n2_parser(spice_n2_parser)
    spice_n1_parser = subparsers.add_parser("spice_n1")
    add_spice_n1_parser(spice_n1_parser)
    # run
    args = parser.parse_args()
    mode = args.mode
    if mode == "conditional_hist":
        del args.mode
        run_conditional_histogram(**args.__dict__)
    elif mode == "cqr":
        del args.mode
        run_cqr(**args.__dict__)
    elif mode == "pcp":
        del args.mode
        run_pcp(**args.__dict__)
    elif mode == "chr":
        del args.mode
        run_chr(**args.__dict__)
    elif mode == "spice_n2":
        del args.mode
        run_spice_n2(**args.__dict__)
    elif mode == "spice_n1":
        del args.mode
        run_spice_n1(**args.__dict__)
