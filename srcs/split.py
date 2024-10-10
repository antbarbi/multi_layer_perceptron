#Train test validation
#https://www.analyticsvidhya.com/blog/2023/11/train-test-validation-split/

import argparse
import pandas as pd
import os

def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('file', type=str, help="Path to the dataset CSV file")
    parser.add_argument('-o', '--output_dir', type=str, help="Relative path to the output of the split function", default=".")
    parser.add_argument('-s', "--seed", type=int, help="Seed for the random split", default=None)

    args = parser.parse_args()

    if not args.file:
        print("Exiting")
        exit(1)
    elif ".csv" != args.file[-4:]:
        print("Exiting: Wrong Format")
        exit(1)
    
    return args.file, args.output_dir, args.seed


def main(dataset: str, output_dir: str, random_seed: int = None) -> None:

    data = pd.read_csv(dataset)

    train_ratio = 0.8
    valid_ratio = 0.2

    if random_seed:
        data_shuffled = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    else:
        data_shuffled = data.sample(frac=1).reset_index(drop=True)

    train_size = int(len(data) * train_ratio)
    train_set = data_shuffled[:train_size]
    validation_set = data_shuffled[train_size:]


    name = os.path.split(dataset)[1]

    train_set.to_csv(f'{output_dir}/Training_{name}', index=False)
    validation_set.to_csv(f'{output_dir}/Validation_{name}', index=False)

if __name__ == "__main__":
    parsed = parser()
    main(*parsed)