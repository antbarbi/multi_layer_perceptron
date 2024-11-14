import argparse
import json
import os
from srcs import split, training, predict

phase_file = ".phase.json"

def phase_selector() -> None:
    parser = argparse.ArgumentParser(description="MLP Program")
    subparsers = parser.add_subparsers(dest="phase", help="Choose between 3 phases of the mlp program: split, training and prediction")
    
    # Create the parser for the "split" phase
    parser_split = subparsers.add_parser("split", help="Split phase")
    parser_split.add_argument('file', type=str, help="Path to the dataset CSV file")
    parser_split.add_argument('-o', '--output_dir', type=str, help="Relative path to the output of the split function", default=".")
    parser_split.add_argument('-s', "--seed", type=int, help="Seed for the random split", default=None)


    # Create the parser for the "training" phase
    parser_training = subparsers.add_parser("training", help="Training phase")
    parser_training.add_argument("--c", "--config_filename", type=str, help="The configuration file for the training phase")
    parser_training.add_argument("--t", "--training_file", type=str, help="The input file for the training phase")
    parser_training.add_argument("--v", "--validation_file", type=str, help="The input file for the validation phase")
    parser_training.add_argument("--o", "--output_filename", type=str, help="The output file for the training phase")

    # Create the parser for the "prediction" phase
    parser_prediction = subparsers.add_parser("prediction", help="Prediction phase")
    parser_prediction.add_argument(
        "--input_filenames", "-i",
        type=str,
        nargs='+',  # Accept one or more filenames
        help="Configuration files for the prediction phase"
    )

    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    args = phase_selector()
    if args.phase == "split":
        split.main(args.file, args.output_dir, args.seed)
    if args.phase == "training":
        training.main(args.t, args.v, args.c, args.o)
    if args.phase == "prediction":
        predict.main(*args.input_filenames)

