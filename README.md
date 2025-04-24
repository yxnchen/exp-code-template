# Experiment Name
Repository for XXX

## Introduction

### Background

### Method and Model

### Main Results

## Repo Structure

- `checkpoints`: saved models' checkpoints
- `data`: put different datasets here
- `figures`: saved figures for presenting in README or paper
- `logs`: logging files
- `models`: layers and models construction
- `utils`: tools for data loading, metrics tracking, logging and training
- `data_inspect.ipynb`: notebook for inspecting the dataset including size, visualization, etc.
- `train.py`: main file to train model
- `test.py`: main file to test model

## Dependencies
```text
python==3.8.18 
torch==1.13.1
numpy==1.22.1
pandas==1.4.3
scipy==1.10.1
```

## Datasets

- Description:
    - 1
    - 2

- Download Link:

## Steps To Run

1. Train
```bash
python train.py
```

Use `python train.py -h` for detail parameter setting.

2. Test
```bash
python test.py
```