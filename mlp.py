import argparse
import json
import os
from srcs import split, training, predict, visualization

phase_file = ".phase.json"

def add_split_subparser(subparsers):
    parser_split = subparsers.add_parser("split", help="Split phase")
    parser_split.add_argument('file', type=str, help="Path to the dataset CSV file")
    parser_split.add_argument('-o', '--output_dir', type=str, default=".",
                              help="Relative path to the output of the split function")
    parser_split.add_argument('-s', "--seed", type=int, default=None,
                              help="Seed for the random split")

def add_training_subparser(subparsers):
    parser_training = subparsers.add_parser("training", help="Training phase")
    parser_training.add_argument("-c", "--config_file", type=str, required=True,
                                  help="The configuration file for the training phase")
    parser_training.add_argument("-t", "--training_file", type=str, required=True,
                                  help="The input file for the training phase")
    parser_training.add_argument("-v", "--validation_file", type=str, required=True,
                                  help="The input file for the validation phase")
    parser_training.add_argument("-o", "--output_file", type=str, required=True,
                                  help="The output file for the training phase")

def add_prediction_subparser(subparsers):
    parser_prediction = subparsers.add_parser("prediction", help="Prediction phase")
    parser_prediction.add_argument("-m", "--model_file", type=str, required=True,
                                   help="The model file for the prediction phase")
    parser_prediction.add_argument("-d", "--data_test", type=str, required=True,
                                   help="The test data file for the prediction phase")

def add_metrics_visualization_subparser(subparsers):
    parser_metrics = subparsers.add_parser("metrics", help="Metrics visualization phase")
    parser_metrics.add_argument(
        'input_filenames',
        type=str,
        nargs='+',
        help="One or more metrics JSON files to visualize"
    )


def phase_selector():
    parser = argparse.ArgumentParser(description="MLP Program")
    subparsers = parser.add_subparsers(dest="phase", required=True,
                                       help="Choose between 3 phases of the MLP program: split, training, prediction and metrics")
    
    add_split_subparser(subparsers)
    add_training_subparser(subparsers)
    add_prediction_subparser(subparsers)
    add_metrics_visualization_subparser(subparsers)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = phase_selector()
    if args.phase == "split":
        split.main(args.file, args.output_dir, args.seed)
    if args.phase == "training":
        training.main(args.training_file, args.validation_file, args.config_file, args.output_file)
    if args.phase == "prediction":
        predict.main(model_file=args.model_file, data_test=args.data_test)
    if args.phase == "metrics":
        visualization.main(*args.input_filenames)


# %%
