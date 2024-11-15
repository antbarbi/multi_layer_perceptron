import os
import argparse
from .model.multi_layer_perceptron import MultiLayerPerceptron



def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.add_argument(
        "--input_filenames", "-i",
        type=str,
        nargs='+',  # Accept one or more filenames
        help="Configuration files for the prediction phase"
    )

    return parser.parse_args()

def check_args(*metric_files: tuple[str]):
    for metric_file in metric_files:
        if not os.path.exists(metric_file):
            raise FileNotFoundError(f"File {metric_file} not found")
        if not metric_file.endswith(".json"):
            raise ValueError("Only .json files are accepted for metrics")


def main(*metric_files: tuple[str]):
    check_args(*metric_files)
    model = MultiLayerPerceptron()
    model.load_metrics(*metric_files)
    model.plot_metrics()


if __name__ == "__main__":
    parsed = parser()
    main(parsed.input_filenames)
