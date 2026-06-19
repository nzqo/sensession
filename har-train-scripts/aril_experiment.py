"""
Put this in the root of:

https://github.com/geekfeiw/ARIL

And run.
"""

import time
import random
import argparse
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import polars as pl
import torch.nn as nn
from tqdm.auto import trange
from models.apl import ResNet as ArilResNet
from models.apl import BasicBlock
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedStratifiedKFold

# ----------------------------------------------------------------------
# Global configuration / paths
# ----------------------------------------------------------------------
MODEL_NAME = "aril"
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    # fmt: off
    "unscaled": Path.cwd() / "data" / "wise_har" / "preprocessed_unscaled_aligned",
    "agcscaled": Path.cwd() / "data" / "wise_har" / "preprocessed_agcscaled_aligned",
    "rms": Path.cwd() / "data" / "wise_har" / "preprocessed_rms_aligned",
    "savgol": Path.cwd() / "data" / "wise_har" / "preprocessed_savgol_aligned",
    "wavelet": Path.cwd() / "data" / "wise_har" / "preprocessed_wavelet_aligned",
    "doublediff": Path.cwd() / "data" / "wise_har" / "preprocessed_doublediff_aligned",
    "morphology": Path.cwd() / "data" / "wise_har" / "preprocessed_morphology_aligned",
    "dbscan": Path.cwd() / "data" / "wise_har" / "preprocessed_dbscan_aligned",
    "ratnam_1": Path.cwd() / "data" / "wise_har" / "preprocessed_ratnam1_aligned",
    "ratnam_2": Path.cwd() / "data" / "wise_har" / "preprocessed_ratnam2_aligned",
    "rollingmed": Path.cwd() / "data" / "wise_har" / "preprocessed_rollingmed_aligned",
    "hampel": Path.cwd() / "data" / "wise_har" / "preprocessed_hampel_aligned",
    "rssi": Path.cwd() / "data" / "wise_har" / "preprocessed_rssi_aligned",
    # fmt: on
}

# ARIL training hyperparameters
NUM_TRAIN_EPOCHS = 200
LEARNING_RATE = 0.005
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 256
SCHED_GAMMA = 0.5

# Bootstrapping settings; used to estimate per-receiver model accuracy
N_SPLITS: int = 5
N_REPEATS: int = 20
N_BOOTSTRAP: int = 5000


# ----------------------------------------------------------------------
# Standardization
# ----------------------------------------------------------------------
class Standardization(Enum):
    """
    Standardization options
    """

    # fmt: off
    NONE                    = ("none",    ())
    MIN_MAX_GLOBAL          = ("minmax",  (0, 1, 2))
    MIN_MAX_SUBCARRIER      = ("minmax",  (0, 1))
    MIN_MAX_WINDOW          = ("minmax",  (1, 2))
    MIN_MAX_ACROSS_EXAMPLES = ("minmax",  (0,))
    ZSCORE_GLOBAL           = ("zscore",  (0, 1, 2))
    ZSCORE_SUBCARRIER       = ("zscore",  (0, 1))
    ZSCORE_WINDOW           = ("zscore",  (1, 2))
    ZSCORE_ACROSS_EXAMPLES  = ("zscore",  (0,))
    # fmt: on

    @property
    def kind(self) -> str:
        """
        Kind of standardization
        """
        return self.value[0]

    @property
    def axes(self) -> tuple[int, ...]:
        """
        Axes along which to apply the standardization
        """
        return self.value[1]


def standardize(
    feature_array: np.ndarray,
    standardization: Standardization,
) -> np.ndarray:
    """
    Apply standardization to array with shape (num_examples, time_steps, num_channels).
    """
    if standardization is Standardization.NONE:
        return feature_array

    axes = standardization.axes

    if standardization.kind == "zscore":
        mean_array = np.mean(feature_array, axis=axes, keepdims=True)
        stddevs = np.std(feature_array, axis=axes, keepdims=True)
        stddevs = np.where(stddevs == 0.0, 1.0, stddevs)
        feature_array = (feature_array - mean_array) / stddevs
    elif standardization.kind == "minmax":
        mins = np.min(feature_array, axis=axes, keepdims=True)
        maxs = np.max(feature_array, axis=axes, keepdims=True)
        val_range = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
        feature_array = (feature_array - mins) / val_range
    else:
        raise ValueError(f"Unsupported standardization kind {standardization.kind}")

    return feature_array


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def set_seeds(seed_value: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_receiver_names(meta_frame: pl.DataFrame) -> list[str]:
    """
    Return receiver names in a stable order.
    """
    return (
        meta_frame.unique("receiver_name", maintain_order=True)
        .get_column("receiver_name")
        .to_list()
    )


def assign_capture_ids(meta_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Attach a capture identifier to each row based on receiver count.
    """
    num_receivers = meta_frame.n_unique("receiver_name")
    capture_identifier_column = (
        pl.arange(0, meta_frame.height) // num_receivers
    ).alias("capture_id")
    return meta_frame.with_columns(capture_identifier_column)


def load_parquet_dataset(dataset_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load CSI and meta data from a dataset directory containing parquet files.
    Expects csi.parquet and meta.parquet in dataset_path.
    """
    csi_frame = pl.read_parquet(dataset_path / "csi.parquet")
    meta_frame = pl.read_parquet(dataset_path / "meta.parquet")
    meta_frame = assign_capture_ids(meta_frame)
    return csi_frame, meta_frame


def csi_abs_to_numpy(csi_frame: pl.DataFrame) -> np.ndarray:
    """
    Convert CSI amplitudes to a dense numpy array.

    Expects a column 'csi_abs' that contains per-packet nested arrays, grouped by meta_id.
    """
    grouped_csi_list = (
        csi_frame.group_by("meta_id", maintain_order=True)
        .agg("csi_abs")
        .get_column("csi_abs")
        .to_list()
    )

    csi_numpy_array = np.array(grouped_csi_list).squeeze()
    return csi_numpy_array  # shape (num_examples=720, window_len=700, subcarrier=56)


def make_xy_from_subset(
    csi_frame: pl.DataFrame,
    meta_subset_frame: pl.DataFrame,
    standardization: Standardization,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature and label arrays for a given subset of meta rows and apply standardization.
    """
    if meta_subset_frame is None or meta_subset_frame.height == 0:
        raise ValueError("Meta subset is empty in make_xy_from_subset")

    joined_frame = csi_frame.join(
        meta_subset_frame.select(["meta_id", "activity_idx"]),
        on="meta_id",
        how="inner",
        maintain_order="left",
    )

    feature_array = csi_abs_to_numpy(joined_frame)

    label_array = (
        joined_frame.group_by("meta_id", maintain_order=True)
        .first()
        .get_column("activity_idx")
        .to_numpy()
        .astype(np.int64)
    )

    feature_array = feature_array.astype(np.float32)
    feature_array = standardize(feature_array, standardization)

    return feature_array, label_array


# ----------------------------------------------------------------------
# Dataloader + train/eval (activity head only)
# ----------------------------------------------------------------------
def build_loader(
    feature_array: np.ndarray,
    label_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for ARIL.

    ARIL ResNet expects input shape (N, C, T)
    feature_array input to this function is expected to be (N, T, C).
    """
    feature_array_copy = np.array(feature_array, copy=True)  # (N, T, C)
    if feature_array_copy.ndim != 3:
        raise ValueError(
            f"Expected feature_array with 3 dims (N, T, C), got {feature_array_copy.shape}"
        )

    # sanity check specific to our dataset
    if feature_array.shape[2] != 56:
        raise ValueError("Expecting (N, T, C) shaped feature array")

    # Transposed to match expected feature dims
    feature_array_copy = np.transpose(feature_array_copy, (0, 2, 1))  # (N, C, T)
    label_array_copy = np.array(label_array, copy=True)

    # To torch
    feature_tensor = torch.from_numpy(feature_array_copy).float()
    label_tensor = torch.from_numpy(label_array_copy).long()
    tensor_dataset = TensorDataset(feature_tensor, label_tensor)

    # Create dataloader
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_once(
    model: nn.Module,
    tensor_loader: DataLoader,
) -> None:
    """
    Train using activity loss only; keep best-accuracy checkpoint (as indicated by their reference test.py script).
    """
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(reduction="sum").to(DEVICE)  # match original
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(0, 10000, 10)),
        gamma=SCHED_GAMMA,
    )

    num_train_instances = len(tensor_loader.dataset)

    epoch_iter = trange(NUM_TRAIN_EPOCHS, desc=MODEL_NAME, leave=False)

    best_acc_act = 0.0
    best_state_dict = None

    for _epoch in epoch_iter:
        model.train()

        epoch_loss_act = 0.0
        correct_train_act = 0

        for samples, labels in tensor_loader:
            samples = samples.to(DEVICE)
            labels_act = labels.to(DEVICE)

            optimizer.zero_grad()

            act_logits, *_ = model(samples)

            # CE with sum reduction like original code.
            loss_act = criterion(act_logits, labels_act)
            loss_act.backward()
            optimizer.step()

            epoch_loss_act += loss_act.item()
            prediction = act_logits.data.max(1)[1]
            correct_train_act += prediction.eq(labels_act.data.long()).sum().item()

        scheduler.step()

        last_loss_act = epoch_loss_act / float(num_train_instances)
        last_acc_act = 100.0 * float(correct_train_act) / float(num_train_instances)

        # track best "checkpoint" by train accuracy
        if last_acc_act > best_acc_act:
            best_acc_act = last_acc_act
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        epoch_iter.set_postfix(
            loss=f"{last_loss_act:.4f}",
            acc=f"{last_acc_act:.2f}%",
        )

    # restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"[{MODEL_NAME}] Best train accuracy (activity): {best_acc_act:.2f}%")


def evaluate(
    model: nn.Module,
    tensor_loader: DataLoader,
) -> float:
    """
    Compute accuracy for a trained model
    """
    model.eval()
    model.to(DEVICE)

    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in tensor_loader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            act_logits, *_ = model(batch_inputs)
            predicted_labels = act_logits.argmax(dim=1)

            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_examples += batch_labels.size(0)

    if total_examples == 0:
        raise ValueError("evaluate called with an empty DataLoader (no examples).")

    return float(correct_predictions) / float(total_examples)


# ----------------------------------------------------------------------
# Cross-receiver experiment
# ----------------------------------------------------------------------
def _run_cross_receiver_collect_records(
    dataset_name: str,
    dataset_path: Path,
    standardization: Standardization,
    seed: int,
) -> list[dict]:
    """
    Core cross-receiver experiment:
    train on each receiver, evaluate on all others,
    return long-format records.
    """
    print(
        f"Starting cross-receiver experiment with ARIL - Train on one, test on others.\n"
        f"  Dataset.........: {dataset_name}\n"
        f"  Standardization.: {standardization.name}\n"
        f"  Base seed.......: {seed}\n"
    )

    set_seeds(seed)

    csi_frame, meta_frame = load_parquet_dataset(dataset_path)
    receiver_names = get_receiver_names(meta_frame)
    num_classes = int(meta_frame["activity_idx"].max()) + 1  # type: ignore[arg-type]

    records: list[dict] = []

    average_time_per_receiver_seconds: float | None = None
    total_receivers = len(receiver_names)
    total_start_time = time.time()

    for receiver_index, train_receiver_name in enumerate(receiver_names):
        receiver_start_time = time.time()
        print(
            f"Training ARIL on receiver {train_receiver_name} "
            f"({receiver_index + 1}/{total_receivers})"
        )

        meta_train_frame = meta_frame.filter(
            pl.col("receiver_name") == train_receiver_name
        )
        feature_array_train, label_array_train = make_xy_from_subset(
            csi_frame,
            meta_train_frame,
            standardization=standardization,
        )

        _, _, num_subcarriers = feature_array_train.shape

        train_loader = build_loader(
            feature_array_train,
            label_array_train,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
        )

        model = ArilResNet(
            block=BasicBlock,
            layers=[1, 1, 1, 1],
            inchannel=num_subcarriers,
            activity_num=num_classes,
            location_num=1,  # present but unused
        )

        train_once(
            model=model,
            tensor_loader=train_loader,
        )

        for eval_receiver_name in receiver_names:
            if eval_receiver_name == train_receiver_name:
                continue

            meta_eval_frame = meta_frame.filter(
                pl.col("receiver_name") == eval_receiver_name
            )
            feature_array_eval, label_array_eval = make_xy_from_subset(
                csi_frame,
                meta_eval_frame,
                standardization=standardization,
            )

            eval_loader = build_loader(
                feature_array_eval,
                label_array_eval,
                batch_size=BATCH_SIZE_EVAL,
                shuffle=False,
            )

            accuracy_value = evaluate(model, eval_loader)
            print(
                f"Trained on {train_receiver_name} "
                f"tested on {eval_receiver_name} "
                f"accuracy {accuracy_value:.4f}"
            )

            records.append(
                {
                    "dataset": dataset_name,
                    "standardization": standardization.name,
                    "train_receiver": train_receiver_name,
                    "test_receiver": eval_receiver_name,
                    "accuracy": float(accuracy_value),
                }
            )

        receiver_elapsed_seconds = time.time() - receiver_start_time

        if average_time_per_receiver_seconds is None:
            average_time_per_receiver_seconds = receiver_elapsed_seconds
        else:
            average_time_per_receiver_seconds = (
                average_time_per_receiver_seconds * receiver_index
                + receiver_elapsed_seconds
            ) / float(receiver_index + 1)

        remaining_receivers = total_receivers - (receiver_index + 1)
        estimated_time_remaining_seconds = (
            remaining_receivers * average_time_per_receiver_seconds
        )

        print(
            f"[{dataset_name} ARIL] "
            f"receiver {receiver_index + 1}/{total_receivers} "
            f"elapsed {receiver_elapsed_seconds / 60.0:.2f} minutes "
            f"estimated remaining {estimated_time_remaining_seconds / 60.0:.2f} minutes"
        )

    total_elapsed_seconds = time.time() - total_start_time
    print(
        f"\nARIL cross-receiver summary: \n"
        f"  Dataset.........: {dataset_name}\n"
        f"  Standardization.: {standardization.name}\n"
        f"  Elapsed time....: {total_elapsed_seconds / 3600.0:.2f} hours\n"
    )

    return records


def run_experiment_cross_receiver(
    dataset_name: str,
    dataset_path: Path,
    standardization: Standardization,
    seed: int = DEFAULT_SEED,
    write_csv: bool = True,
) -> pl.DataFrame:
    """
    Cross-receiver experiment
    """
    records = _run_cross_receiver_collect_records(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        standardization=standardization,
        seed=seed,
    )

    cross_receiver_frame = pl.DataFrame(records)

    if write_csv:
        output_directory = Path("results")
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_directory
            / f"{MODEL_NAME}_{dataset_name}_{standardization.name}_cross.csv"
        )
        cross_receiver_frame.write_csv(output_path)
        print(f"Saved ARIL cross receiver results to {output_path}")

    return cross_receiver_frame


# ----------------------------------------------------------------------
# Standardization ablation (cross-receiver over multiple settings)
# ----------------------------------------------------------------------
def run_std_ablation_cross_receiver(
    dataset_name: str,
    dataset_path: Path,
    seed: int = DEFAULT_SEED,
) -> Path:
    """
    Run cross receiver experiments over multiple standardizations and collect results.
    """
    all_records: list[dict] = []
    ablation_stds = list(Standardization)

    for standardization in ablation_stds:
        all_records.extend(
            _run_cross_receiver_collect_records(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                standardization=standardization,
                seed=seed,
            )
        )

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_directory / f"{MODEL_NAME}_{dataset_name}_cross_std_ablation.csv"
    )
    pl.DataFrame(all_records).write_csv(output_path)
    print(f"Saved ARIL cross standardization ablation to {output_path}")
    return output_path


# ----------------------------------------------------------------------
# Dataset ablation (cross-receiver over multiple datasets)
# NOTE: This is our agc-norms ablation. However, we preprocess the CSI
# with the different AGC steps using our preprocess_align.py script
# BEFOREHAND. That's why here we ablate over datasets.
# ----------------------------------------------------------------------
def run_dataset_ablation_cross_receiver(
    standardization: Standardization,
    seed: int = DEFAULT_SEED,
) -> Path:
    """
    Run cross-receiver experiments over all DATASETS for a fixed standardization
    (e.g., ZSCORE_ACROSS_EXAMPLES) and collect results in one CSV.
    """
    all_records: list[dict] = []

    for dataset_name, dataset_path in DATASETS.items():
        all_records.extend(
            _run_cross_receiver_collect_records(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                standardization=standardization,
                seed=seed,
            )
        )

    output_directory = Path("results")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_directory
        / f"{MODEL_NAME}_{standardization.name}_cross_dataset_ablation.csv"
    )
    pl.DataFrame(all_records).write_csv(output_path)
    print(f"Saved ARIL cross dataset ablation to {output_path}")
    return output_path


# ----------------------------------------------------------------------
# K-fold experiment
# ----------------------------------------------------------------------
def run_experiment_kfold_aril(
    dataset_name: str,
    dataset_path: Path,
    standardization: Standardization,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[float]]:
    """
    Run repeated stratified k-fold experiments per receiver (activity-only loss).
    """
    print(
        f"Running k-fold ARIL experiment to estimate model accuracies per receiver:\n"
        f"  Total runs......: {N_SPLITS} splits x {N_REPEATS} repeats\n"
        f"  Dataset.........: {dataset_name}\n"
        f"  Standardization.: {standardization.name}\n"
        f"  Base seed.......: {seed}\n"
    )

    set_seeds(seed)
    random_generator = np.random.default_rng(seed)

    csi_frame, meta_frame = load_parquet_dataset(dataset_path)
    receiver_names = get_receiver_names(meta_frame)
    num_classes = int(meta_frame["activity_idx"].max()) + 1  # type: ignore[arg-type]

    meta_per_capture_frame = (
        meta_frame.group_by("capture_id")
        .agg(pl.first("activity_idx"))
        .sort("capture_id")
    )

    capture_identifier_list = meta_per_capture_frame["capture_id"].to_list()
    capture_label_list = meta_per_capture_frame["activity_idx"].to_list()

    kfold_splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=seed,
    )

    fold_results: dict[str, list[float]] = {
        receiver_name: [] for receiver_name in receiver_names
    }
    total_folds = N_SPLITS * N_REPEATS

    total_start_time = time.time()
    average_fold_time_seconds = None

    for fold_index, (train_indices, test_indices) in enumerate(
        kfold_splitter.split(capture_identifier_list, capture_label_list)
    ):
        print(f"Starting on fold {fold_index + 1}/{total_folds}")
        fold_start_time = time.time()
        set_seeds(seed + fold_index)

        # Split dataframes in train/test
        train_ids = [capture_identifier_list[index] for index in train_indices]
        test_ids = [capture_identifier_list[index] for index in test_indices]
        meta_train_frame = meta_frame.filter(pl.col("capture_id").is_in(train_ids))
        meta_test_frame = meta_frame.filter(pl.col("capture_id").is_in(test_ids))

        for receiver_name in receiver_names:
            print(f"\tReceiver {receiver_name}")

            # fmt: off
            meta_train_receiver_frame = meta_train_frame.filter(pl.col("receiver_name") == receiver_name)
            meta_test_receiver_frame = meta_test_frame.filter(pl.col("receiver_name") == receiver_name)
            # fmt: on

            feature_array_train, label_array_train = make_xy_from_subset(
                csi_frame,
                meta_train_receiver_frame,
                standardization=standardization,
            )
            feature_array_test, label_array_test = make_xy_from_subset(
                csi_frame,
                meta_test_receiver_frame,
                standardization=standardization,
            )

            _, _, num_subcarriers = feature_array_train.shape

            train_loader = build_loader(
                feature_array_train,
                label_array_train,
                batch_size=BATCH_SIZE_TRAIN,
                shuffle=True,
            )

            test_loader = build_loader(
                feature_array_test,
                label_array_test,
                batch_size=BATCH_SIZE_EVAL,
                shuffle=False,
            )

            model = ArilResNet(
                block=BasicBlock,
                layers=[1, 1, 1, 1],
                inchannel=num_subcarriers,
                activity_num=num_classes,
                location_num=1,
            )

            train_once(
                model=model,
                tensor_loader=train_loader,
            )

            fold_accuracy = evaluate(model, test_loader)
            fold_results[receiver_name].append(fold_accuracy)
            print(f"\tTest accuracy for this fold {fold_accuracy:.4f}")

        fold_elapsed_seconds = time.time() - fold_start_time

        if average_fold_time_seconds is None:
            average_fold_time_seconds = fold_elapsed_seconds
        else:
            average_fold_time_seconds = (
                average_fold_time_seconds * fold_index + fold_elapsed_seconds
            ) / float(fold_index + 1)

        remaining_folds = total_folds - (fold_index + 1)
        estimated_time_remaining_seconds = remaining_folds * average_fold_time_seconds

        print(
            f"ARIL fold summary: [on {dataset_name}]\n"
            f"fold................: {fold_index + 1}/{total_folds}\n"
            f"elapsed.............: {fold_elapsed_seconds / 60.0:.2f} minutes\n"
            f"estimated remaining.: {estimated_time_remaining_seconds / 60.0:.2f} minutes"
        )

    total_elapsed_seconds = time.time() - total_start_time
    print(
        f"ARIL kfold: \n"
        f"  Dataset.........: {dataset_name}\n"
        f"  Standardization.: {standardization.name}\n"
        f"  Elapsed time....: {total_elapsed_seconds / 3600.0:.2f} hours"
    )

    raw_record_list = []
    for receiver_name, accuracy_list in fold_results.items():
        for fold_id, accuracy_value in enumerate(accuracy_list):
            raw_record_list.append(
                {
                    "receiver": receiver_name,
                    "fold": fold_id,
                    "accuracy": float(accuracy_value),
                }
            )

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    raw_output_path = (
        output_directory
        / f"{MODEL_NAME}_{dataset_name}_{standardization.name}_kfold_raw.csv"
    )
    pl.DataFrame(raw_record_list).write_csv(raw_output_path)

    summary_record_list = []
    for receiver_name, accuracy_list in fold_results.items():
        if len(accuracy_list) == 0:
            continue

        accuracy_array = np.array(accuracy_list, dtype=np.float32)
        mean_accuracy = float(accuracy_array.mean())
        standard_deviation_accuracy = float(accuracy_array.std(ddof=0))

        bootstrap_means = np.array(
            [
                random_generator.choice(
                    accuracy_array,
                    size=accuracy_array.size,
                    replace=True,
                ).mean()
                for _ in range(N_BOOTSTRAP)
            ],
            dtype=np.float32,
        )

        confidence_interval_95_lower, confidence_interval_95_upper = np.percentile(
            bootstrap_means,
            [2.5, 97.5],
        )
        confidence_interval_99_lower, confidence_interval_99_upper = np.percentile(
            bootstrap_means,
            [0.5, 99.5],
        )

        summary_record_list.append(
            {
                "receiver": receiver_name,
                "mean_accuracy": mean_accuracy,
                "std": standard_deviation_accuracy,
                "ci95_lower": float(confidence_interval_95_lower),
                "ci95_upper": float(confidence_interval_95_upper),
                "ci99_lower": float(confidence_interval_99_lower),
                "ci99_upper": float(confidence_interval_99_upper),
            }
        )

        print(
            f"Per-receiver k-fold acc summary:\n"
            f"  Receiver...............: {receiver_name}\n"
            f"  Mean accuracy..........: {mean_accuracy:.4f}\n"
            f"  95 percent CI interval.: [{confidence_interval_95_lower:.4f}, {confidence_interval_95_upper:.4f}]\n"
            f"  99 percent CI interval.: [{confidence_interval_99_lower:.4f}, {confidence_interval_99_upper:.4f}]"
        )

    summary_output_path = (
        output_directory
        / f"{MODEL_NAME}_{dataset_name}_{standardization.name}_kfold_summary.csv"
    )
    pl.DataFrame(summary_record_list).write_csv(summary_output_path)

    return fold_results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    """
    Entry point for running ARIL experiments from the command line.

      --mode cross     : cross-receiver experiments
      --mode kfold     : repeated stratified k-fold per receiver
      --mode ablation  : cross-receiver over all standardization settings

    Default behavior if nothing specified: run ablation on both datasets.
    """
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Dataset key to use from the DATASETS mapping (global in module).",
    )

    argument_parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["cross", "kfold", "std_ablation", "agc_ablation"],
        help="Experiment mode: cross receiver, k fold, or standardization ablation.",
    )

    argument_parser.add_argument(
        "--std",
        type=str,
        default=None,
        choices=[c.name for c in Standardization],
        help="Standardization setting name for example NONE or ZSCORE_ACROSS_EXAMPLES.",
    )

    argument_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for all experiments.",
    )

    parsed_arguments = argument_parser.parse_args()

    if parsed_arguments.dataset and parsed_arguments.mode:
        dataset_name = parsed_arguments.dataset
        dataset_path = DATASETS[dataset_name]

        if parsed_arguments.mode == "std_ablation":
            if parsed_arguments.std is not None:
                print("[warning] --std is ignored when --mode std_ablation is used.")

            run_std_ablation_cross_receiver(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                seed=parsed_arguments.seed,
            )
            return

        # Get desired standardization (default to feature-wise zscore across examples)
        if parsed_arguments.std is not None:
            standardization = Standardization[parsed_arguments.std]
        else:
            standardization = Standardization.ZSCORE_ACROSS_EXAMPLES

        if parsed_arguments.mode == "agc_ablation":
            run_dataset_ablation_cross_receiver(
                standardization=standardization,
                seed=parsed_arguments.seed,
            )
            return

        # Cross-device experiment: Train on one, test on other receivers and monitor accuracy
        if parsed_arguments.mode == "cross":
            run_experiment_cross_receiver(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                standardization=standardization,
                seed=parsed_arguments.seed,
            )
            return

        # Per-device experiment: Get proper statistical estimates of how well a receiver performs
        if parsed_arguments.mode == "kfold":
            run_experiment_kfold_aril(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                standardization=standardization,
                seed=parsed_arguments.seed,
            )
            return

    # Default behavior if nothing is specified: run ablation on both datasets

    for dataset_name, dataset_path in DATASETS.items():
        run_std_ablation_cross_receiver(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )


if __name__ == "__main__":
    main()
