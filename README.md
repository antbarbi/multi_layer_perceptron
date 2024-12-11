# Multi-Layer Perceptron (MLP) Program

This project provides a command-line interface (CLI) for managing different phases of a Multi-Layer Perceptron (MLP) workflow, including data splitting, training, prediction, and metrics visualization.

## Requirements

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

## Usage

The `mlp.py` script supports four main phases: `split`, `training`, `prediction`, and `metrics`. Each phase has its own set of arguments.

### Split Phase

The `split` phase splits a dataset into training and validation sets. 
```sh
py mlp.py split {filename} -o {output_dir} -s {seeding}
```

### Training Phase
The `training` phase trains a model on the data provided.
```sh
py mlp.py training -c {config_file} -t {training_dataset} -v {validation_dataset} -o {metrics_output_filename}
```

### Prediction Phase

```sh
py mlp.py prediction -m {model_file} -d {dataset}
```

### Visualization Phase

```sh
py mlp.py metrics {metrics_file}, {metrics_file_2}...
```
