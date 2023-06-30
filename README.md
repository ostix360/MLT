[![CodeQL](https://github.com/ostix360/MLT/actions/workflows/codeql.yml/badge.svg?event=push)](https://github.com/ostix360/MLT/actions/workflows/codeql.yml)

# Multiple Lora Training

## Introduction

This project demonstrates how to train multiple Lora and how it can be efficient.



## Getting Started

### Requirements

-   Python 3.x


### Installation

1. Clone the repository

```bash
git clone https://github.com/ostix360/MLT.git
```

2. Navigate to the project directory

```bash
cd MLT
```

3. Install packages

```bash
pip install -r requirements.txt
```
       

### Usage

Create your python script and import the library


Look at the [example](https://github.com/ostix360/MLT/blob/master/example.py) to see how to use the library.


## Evaluation 

The example model is train with the adafactor optimizer contrary to others model that use AdamW.
An other difference is that the example model is train with split dataset.

The steps for training:
- 1st training step it the finetuning step (finetune_first=True) with the rotten tomatoes dataset

- 2nd training step is the training of one lora (called sst2) with the lora config and the sst2 dataset

- 3rd training step is the training of the sst2 lora with 50% (train_ratio=0.5) of both of the rotten tomatoes and sst2 datasets

- 4th training step is the training of one lora (called imdb) with the lora config and the imdb dataset with the sst2 lora loaded but not as trainable

- 5th training step is the training of the sst2 and imdb lora with 50% (train_ratio=0.5) of all of the rotten tomatoes, sst2 and imdb datasets

Each step correspond to an epoch.

### Other models found in hub

| Dataset  |                            rotten tomatoes                            |                       sst2                       |                               imdb                               |
|:--------:|:---------------------------------------------------------------------:|:------------------------------------------------:|:----------------------------------------------------------------:|
| Accuracy |                                 0.83                                  |                      0.836                       |                               0.86                               |
|   Loss   |                                 0.801                                 |                       1.02                       |                              0.4535                              |
|  Epoch   |                                  1.0                                  |                       1.0                        |                               1.29                               |
|   link   | [link](https://huggingface.co/flowfree/bert-finetuned-rottentomatoes) | [link](https://huggingface.co/ostix360/MLT-sst2) | [link](https://huggingface.co/fabriceyhc/bert-base-uncased-imdb) |

### The Model finetuned : 

Accuracy and loss during training steps

| Steps |   1   |   2   |   3   |   4   |   5   |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  Acc  | 0.847 | 0.931 | 0.86  | 0.875 | 0.872 |
| Loss  | 0.382 | 0.285 | 0.279 | 0.314 | 0.279 |



