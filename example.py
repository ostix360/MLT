import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as TARGET_MODULES_MAPPING
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding

import utils
from mlt import MLLoader
from mlt.MLTrainer import MLTrainer

# Load Model

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, return_dict=True)

# Load Datasets
# Using split datasets because of low hardware resources

rotten_tomatoes_datasets_t = load_dataset("rotten_tomatoes", split="train[:100%]").shuffle()
sst2_datasets_t = load_dataset("sst2", split="train[:75%]").shuffle()
imdb_datasets_t = load_dataset("imdb", split="train[:6%]+train[92%:]").shuffle()

rotten_tomatoes_datasets_v = load_dataset("rotten_tomatoes", split="test[:100%]").shuffle()
sst2_datasets_v = load_dataset("sst2", split="validation[:50%]").shuffle()
imdb_datasets_v = load_dataset("imdb", split="test[:3%]+test[97%:]").shuffle()


# Tokenize Datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


def tokenize_function_sst2(examples):
    return tokenizer(examples["sentence"], truncation=True)


sst2_datasets_t = sst2_datasets_t.map(tokenize_function_sst2, batched=True)
rotten_tomatoes_datasets_t = rotten_tomatoes_datasets_t.map(tokenize_function, batched=True)
imdb_datasets_t = imdb_datasets_t.map(tokenize_function, batched=True)

sst2_datasets_v = sst2_datasets_v.map(tokenize_function_sst2, batched=True)
rotten_tomatoes_datasets_v = rotten_tomatoes_datasets_v.map(tokenize_function, batched=True)
imdb_datasets_v = imdb_datasets_v.map(tokenize_function, batched=True)

# Create Datasets for MLTrainer
# Same key names for both datasets train and test
train_ds = {"rotten_tomatoes": rotten_tomatoes_datasets_t.rename_column("label", "labels").remove_columns(["text"]),
            "sst2": sst2_datasets_t.rename_column("label", "labels").remove_columns(["sentence", "idx"]),
            "imdb": imdb_datasets_t.rename_column("label", "labels").remove_columns(["text"])}
test_ds = {"rotten_tomatoes": rotten_tomatoes_datasets_v.rename_column("label", "labels").remove_columns(["text"]),
           "sst2": sst2_datasets_v.rename_column("label", "labels").remove_columns(["sentence", "idx"]),
           "imdb": imdb_datasets_v.rename_column("label", "labels").remove_columns(["text"])}


# Compute Metrics for MLTrainer and more specifically for the transformers Trainer
def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    computed_metrics = metric.compute(predictions=predictions, references=labels)
    print(computed_metrics)
    return computed_metrics


# Training Arguments for each training part
training_args = TrainingArguments("./test_trainer",
                                  logging_steps=20,
                                  num_train_epochs=1,   # Each dataset will be trained for 1 epoch
                                  remove_unused_columns=False,
                                  learning_rate=3e-5,
                                  save_total_limit=1,
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=4,
                                  gradient_checkpointing=True,
                                  optim="adafactor")

# Data Collator for MLTrainer
data_collator = DataCollatorWithPadding(tokenizer)

# Lora config for all Loras added to the model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES_MAPPING.get("bert"),
    lora_dropout=0.06,
    modules_to_save=["classifier", "score"],    # Modules added because it's a model for sequence classification
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# Create MLTrainer from parameters defined above
trainer = MLTrainer(model=model,
                    finetune_first=True,
                    training_args=training_args,
                    train_datasets=train_ds,
                    eval_datasets=test_ds,
                    data_collator=data_collator,
                    lora_config=lora_config,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                    train_ratio=0.5,    # 50% of the dataset will be used for the multiple lora training part
                    )

# 1st training step it the finetuning step (finetune_first=True) with the rotten tomatoes dataset

# 2nd training step is the training of one lora (called sst2) with the lora config and the sst2 dataset

# 3rd training step is the training of the sst2 lora with 50% (train_ratio=0.5)
# of both of the rotten tomatoes and sst2 datasets

# 4th training step is the training of one lora (called imdb) with the lora config
# and the imdb dataset with the sst2 lora loaded but not as trainable

# 5th training step is the training of the sst2 and imdb lora with 50% (train_ratio=0.5)
# of all of the rotten tomatoes, sst2 and imdb datasets

# Finally model and loras are saved in the ./test_trainer folder


# Prepare custom training loop
optimizer = AdamW(model.parameters(), lr=4e-5)
device = utils.get_device()


def train(model, train_dataloader, eval_dataloader):    # Custom training loop
    num_epochs = 1
    model.to(device)
    lr_scheduler, num_training_steps = utils.get_lr_scheduler(optimizer, train_dataloader, num_epochs)
    utils.train_model(model, train_dataloader, num_epochs, optimizer, lr_scheduler, device, num_training_steps)
    print(utils.evaluate_model(model, eval_dataloader, device))


# trainer.custom_train(train)   # Uncomment to train with custom training loop
# trainer.train()   # Uncomment to train with the transformers Trainer

# Load the finetuned model
eval_model = AutoModelForSequenceClassification.from_pretrained("./test_trainer", return_dict=True)

c_train_dataset = {
    k: DataLoader(
        v.shuffle(),
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    for k, v in train_ds.items()
}

c_eval_dataset = {
    k: DataLoader(
        v.shuffle(),
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    for k, v in test_ds.items()
}


eval_MLmodel = MLLoader.loadMLModel(eval_model, ["sst2", "imdb"], "./test_trainer")  # load loras locally


def accuracy(model, ):
    model.to(device)
    print(utils.evaluate_model(model, c_eval_dataset["rotten_tomatoes"], device))   # 0.847 accuracy
    print(utils.evaluate_model(model, c_eval_dataset["sst2"], device))  # 0.931 accuracy
    print(utils.evaluate_model(model, c_eval_dataset["imdb"], device))  # 0.874 accuracy


# Evaluate the model on the 3 datasets
accuracy(eval_MLmodel)
