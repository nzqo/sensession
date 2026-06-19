"""
Convert parquet-backed CSI recordings to the NPY backend format.

Recursively searches a root directory for folders that contain both
``csi.parquet`` and ``meta.parquet``, then converts each one in-place:

* Reads every (receiver, meta_id) group from ``csi.parquet``
* Reconstructs complex CSI as ``csi_abs * exp(1j * csi_phase)``
* Writes one ``.npz`` file per group under ``<recording>/csi/``
* Leaves ``meta.parquet`` untouched (same format for both backends)
* Optionally removes ``csi.parquet`` afterwards (``--delete``)

Usage
-----
    python scripts/parquet_to_numpy.py /path/to/data
    python scripts/parquet_to_numpy.py /path/to/data --delete
    python scripts/parquet_to_numpy.py /path/to/data --dry-run
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


# -------------------------------------------------------------------------------------
# -- Conversion helpers
# -------------------------------------------------------------------------------------
def _find_recordings(root: Path) -> list[Path]:
    """
    Recursively find all directories that contain both csi.parquet and meta.parquet.

    Args:
        root : Directory to search from.

    Returns:
        Sorted list of matching directory paths.
    """
    matches = [
        p.parent
        for p in root.rglob("csi.parquet")
        if (p.parent / "meta.parquet").is_file()
    ]
    return sorted(matches)


def _already_converted(recording: Path) -> bool:
    """
    Check whether a recording directory has already been converted to NPY format.

    A directory is considered converted when a ``csi/`` subdirectory is present,
    regardless of whether ``csi.parquet`` still exists.

    Args:
        recording : Recording directory to check.
    """
    return (recording / "csi").is_dir()


def _write_group(
    group_df: pl.DataFrame,
    meta_id_str: str,
    receiver_name: str,
    csi_dir: Path,
):
    """
    Write one (receiver, meta_id) group from a csi DataFrame to a .npz file.

    Args:
        group_df      : Rows belonging to a single (receiver, meta_id) pair.
        meta_id_str   : String representation of the meta_id.
        receiver_name : Name of the receiver device.
        csi_dir       : Directory to write the .npz file into.
    """
    timestamps = group_df.get_column("timestamp").cast(pl.UInt64).to_numpy()
    seqnums = group_df.get_column("sequence_number").to_numpy()

    # Reconstruct complex CSI: (n_frames, n_ant, n_streams, n_sub)
    abs_stack = np.array(group_df.get_column("csi_abs").to_list(), dtype=np.float64)
    phase_stack = np.array(group_df.get_column("csi_phase").to_list(), dtype=np.float64)
    csi = abs_stack * np.exp(1j * phase_stack)

    fname = f"{receiver_name}_{meta_id_str}.npz"
    np.savez(csi_dir / fname, csi=csi, timestamps=timestamps, sequence_numbers=seqnums)
    logger.debug(f"  wrote csi/{fname}  shape={csi.shape}")


def _convert_recording(recording: Path, *, delete_parquet: bool) -> int:
    """
    Convert a single parquet-backed recording to the NPY format.

    Args:
        recording      : Recording directory containing csi.parquet and meta.parquet.
        delete_parquet : When True, remove ``csi.parquet`` after a successful conversion.

    Returns:
        Number of (receiver, meta_id) groups written.
    """
    csi_parquet = recording / "csi.parquet"
    csi_dir = recording / "csi"

    logger.info(f"Converting: {recording}")
    csi_dir.mkdir(exist_ok=True)

    csi_df = pl.read_parquet(csi_parquet)
    meta_df = pl.read_parquet(recording / "meta.parquet")

    # Build a fast meta_id → receiver_name lookup
    receiver_map: dict[str, str] = dict(
        meta_df.select("meta_id", "receiver_name").iter_rows()
    )

    groups_written = 0
    for (meta_id,), group_df in csi_df.group_by("meta_id", maintain_order=True):
        meta_id_str = str(meta_id)
        _write_group(
            group_df, meta_id_str, receiver_map.get(meta_id_str, "unknown"), csi_dir
        )
        groups_written += 1

    if delete_parquet:
        csi_parquet.unlink()
        logger.info(f"  removed {csi_parquet.name}")

    return groups_written


# -------------------------------------------------------------------------------------
# -- CLI entry point
# -------------------------------------------------------------------------------------
def main():
    """
    Parse arguments and run the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert parquet CSI recordings to the NPY backend format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to search for parquet recordings.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Remove csi.parquet after successful conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only list recordings that would be converted; do not write anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level output (prints every file written).",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if not args.root.is_dir():
        logger.error(f"Root path does not exist or is not a directory: {args.root}")
        sys.exit(1)

    recordings = _find_recordings(args.root)

    if not recordings:
        logger.warning(f"No parquet recordings found under {args.root}")
        sys.exit(0)

    logger.info(f"Found {len(recordings)} recording(s) under {args.root}")

    already_done = [r for r in recordings if _already_converted(r)]
    to_convert = [r for r in recordings if not _already_converted(r)]

    if already_done:
        logger.info(f"Skipping {len(already_done)} already-converted recording(s):")
        for r in already_done:
            logger.info(f"  (skip) {r}")

    if not to_convert:
        logger.info("Nothing left to convert.")
        sys.exit(0)

    if args.dry_run:
        logger.info("Dry run — would convert:")
        for r in to_convert:
            logger.info(f"  {r}")
        sys.exit(0)

    total_groups = 0
    total_failed = 0

    for recording in to_convert:
        try:
            n = _convert_recording(recording, delete_parquet=args.delete)
            total_groups += n
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to convert {recording}: {exc}")
            total_failed += 1

    logger.info(
        f"Done. Converted {len(to_convert) - total_failed}/{len(to_convert)} recording(s), "
        f"{total_groups} group(s) written."
        + (f" {total_failed} failure(s)." if total_failed else "")
    )

    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
