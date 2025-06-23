import argparse
import os
import time
import pandas as pd

try:
    from lips.dataset.airfransDataSet import AirfRANSDataSet
except Exception as e:  # pragma: no cover - optional dependency may not exist
    raise ImportError("The lips package is required to load AirfRANS datasets") from e

ATTR_NAMES = (
    'x-position',
    'y-position',
    'x-inlet_velocity',
    'y-inlet_velocity',
    'distance_function',
    'x-normals',
    'y-normals',
    'x-velocity',
    'y-velocity',
    'pressure',
    'turbulent_viscosity',
)
ATTR_X = ATTR_NAMES[:7]
ATTR_Y = ATTR_NAMES[7:]


def load_dataset(directory: str, task: str) -> AirfRANSDataSet:
    """Load the AirfRANS dataset for the given task."""
    ds = AirfRANSDataSet(
        config=None,
        name=f"airfrans_{task}",
        task=task,
        split="training",
        attr_names=ATTR_NAMES,
        attr_x=ATTR_X,
        attr_y=ATTR_Y,
        log_path="exploration_log",
    )
    start = time.time()
    ds.load(path=directory)
    print(f"Dataset '{task}' loaded in {time.time() - start:.1f}s")
    return ds


def compute_statistics(ds: AirfRANSDataSet) -> pd.DataFrame:
    """Return a DataFrame with descriptive statistics for the dataset."""
    df = pd.DataFrame({a: ds.data[a] for a in ATTR_NAMES if a in ds.data})
    stats = df.describe().round(4)
    print(stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore AirfRANS dataset")
    parser.add_argument('--path', default='Dataset', help='dataset directory')
    parser.add_argument('--task', default='scarce', choices=['scarce', 'full'])
    parser.add_argument('--outfile', default='airfrans_stats.csv')
    args = parser.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('OMP_NUM_THREADS', '2')

    ds = load_dataset(args.path, args.task)
    stats = compute_statistics(ds)
    stats.to_csv(args.outfile)
    print(f"Statistics saved to {args.outfile}")


if __name__ == '__main__':  # pragma: no cover - CLI entry
    main()
