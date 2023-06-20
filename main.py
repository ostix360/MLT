import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as target_modules_mapping
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding

import utils
from MLTrainer import MLTrainer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sst2_datasets_t = load_dataset("sst2", split="train[:5%]")
rotten_tomatoes_datasets_t = load_dataset("rotten_tomatoes", split="train[:20%]")
imdb_datasets_t = load_dataset("imdb", split="train[:10%]")

sst2_datasets_v = load_dataset("sst2", split="validation[:10%]")
rotten_tomatoes_datasets_v = load_dataset("rotten_tomatoes", split="test[:10%]")
imdb_datasets_v = load_dataset("imdb", split="test[:10%]")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


def tokenize_function_sst2(examples):
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


sst2_datasets_t = sst2_datasets_t.map(tokenize_function_sst2, batched=True)
rotten_tomatoes_datasets_t = rotten_tomatoes_datasets_t.map(tokenize_function, batched=True)
imdb_datasets_t = imdb_datasets_t.map(tokenize_function, batched=True)

sst2_datasets_v = sst2_datasets_v.map(tokenize_function_sst2, batched=True)
rotten_tomatoes_datasets_v = rotten_tomatoes_datasets_v.map(tokenize_function, batched=True)
imdb_datasets_v = imdb_datasets_v.map(tokenize_function, batched=True)

train_ds = {"sst2": sst2_datasets_t.rename_column("label", "labels").remove_columns(["sentence", "idx"]),
            "rotten_tomatoes": rotten_tomatoes_datasets_t.rename_column("label", "labels").remove_columns(["text"]),
            "imdb": imdb_datasets_t.rename_column("label", "labels").remove_columns(["text"])}
test_ds = {"sst2": sst2_datasets_v.rename_column("label", "labels").remove_columns(["sentence", "idx"]),
           "rotten_tomatoes": rotten_tomatoes_datasets_v.rename_column("label", "labels").remove_columns(["text"]),
           "imdb": imdb_datasets_v.rename_column("label", "labels").remove_columns(["text"])}

training_args = TrainingArguments("test_trainer",
                                  logging_steps=20,
                                  num_train_epochs=1,
                                  remove_unused_columns=False,
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=4,
                                  gradient_checkpointing=True,
                                  optim="adafactor")

data_collator = DataCollatorWithPadding(tokenizer)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules_mapping.get("bert").append("classifier"),
    lora_dropout=0.06,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

trainer = MLTrainer(model=model,
                    finetune_first=True,
                    training_args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    lora_config=lora_config,
                    compute_metrics=compute_metrics, )

optimizer = AdamW(model.parameters(), lr=3e-5)
device = utils.get_device()


def train(model, train_dataloader, eval_dataloader):
    num_epochs = 1
    model.to(device)
    lr_scheduler, num_training_steps = utils.get_lr_scheduler(optimizer, train_dataloader, num_epochs)
    # utils.train_model(model, train_dataloader, num_epochs, optimizer, lr_scheduler, device, num_training_steps)
    utils.evaluate_model(model, eval_dataloader, device)


# trainer.custom_train(train)
# trainer.train()
