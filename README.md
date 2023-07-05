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

import MLT and create the constructor of MLTrainer:

```python
from MLT import MLTrainer

trainer = MLTrainer(model=model, # model to train
                    finetune_first=True, # if True, the first training step will finetune the model with the first dataset
                    training_args=training_args, # training args from transformers
                    train_datasets=train_ds, # dict of datasets for training
                    eval_datasets=test_ds, # dict of datasets for evaluation
                    data_collator=data_collator, # data collator from transformers
                    lora_config=lora_config, # lora config for all lora that will be trained
                    tokenizer=tokenizer, # tokenizer from transformers
                    compute_metrics=compute_metrics, # compute metrics for transformers' trainer
                    loras=[], # list of lora pretrained that will be loaded and trained if their names are in the train_datasets
                    optimizer=None, # optimizer for transformers' trainer
                    train_ratio=0.5,    # 50% of the dataset will be used for the multiple lora training part
                 )
```

Then train the model with the train method:

```python
trainer.train() # train the model
# or
trainer.custom_train(trainer=custom_loop) # train the model with custom training loop
```

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


### T5 translation fine-tunig

Evaluation of the t5-small model trained on the entire opus books dataset.
This model has only 60M parameters.

As the previous model the t5-small model is fine-tuned with MLT method.



The table bellow shows the blue score and the loss of the model for each training step.
The step 1 is the training of the de-en lora with the de-en dataset.
The step 2 is the same but with 50% of the de-en dataset.
The step 3 is the training of the en-de lora with the de-en dataset swapped (so en-de).
The step 4 is the training of the de-en, en-de lora with 50% of the de-en and en-de datasets.
And so on...

After the step 4 the model has 62M parameters.

|    Steps    | 1 de-en | 2 mix (1) | 3 en-de | 4 mix (2-3) |   5   |
|:-----------:|:-------:|:---------:|:-------:|:-----------:|:-----:|
| Blue Before |  0.582  |   7.433   | 10.785  |    7.868    | 0.000 |
| Blue After  |  7.433  |  10.433   | 14.010  |    12.19    | 0.000 |
|    Loss     |   3.1   |   2.99    |  2.41   |    2.75     | 0.000 |


