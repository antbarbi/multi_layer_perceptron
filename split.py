#Train test validation
#https://www.analyticsvidhya.com/blog/2023/11/train-test-validation-split/

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()

parser.add_argument('file', type=str, help="Path to the dataset CSV file")
parser.add_argument('-s', "--seed", type=int, help="Seed for the random split")

args = parser.parse_args()

if not args.file:
    print("Exiting")
    exit()
elif ".csv" != args.file[-4:]:
    print("Exiting: Wrong Format")
    exit()

data = pd.read_csv(args.file)

train_ratio = 0.8
valid_ratio = 0.2

data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(data) * train_ratio)
train_set = data_shuffled[:train_size]
validation_set = data_shuffled[train_size:]


name = os.path.split(args.file)[1]

train_set.to_csv(f'Training_{name}', index=False)
validation_set.to_csv(f'Validation_{name}', index=False)