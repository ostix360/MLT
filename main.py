import numpy as np
from datasets import load_dataset, load_metric
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, DistilBertTokenizerFast
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as target_modules_mapping

from MLTrainer import MLTrainer

checkpoint = "michelecafagna26/gpt2-medium-finetuned-sst2-sentiment"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sst2_datasets = load_dataset("sst2", split="train[:10%]+test[:10%]")
rotten_tomatoes_datasets = load_dataset("rotten_tomatoes", split="train[:10%]+test[:10%]")
imdb_datasets = load_dataset("imdb", split="train[:10%]+test[:10%]")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


def tokenize_function_sst2(examples):
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


sst2_datasets = sst2_datasets.map(tokenize_function_sst2, batched=True)
rotten_tomatoes_datasets = rotten_tomatoes_datasets.map(tokenize_function, batched=True)
imdb_datasets = imdb_datasets.map(tokenize_function, batched=True)

train_ds = {"sst2": sst2_datasets["train"].rename_column("label", "labels").remove_columns(["sentence", "idx"]),
            "rotten_tomatoes": rotten_tomatoes_datasets["train"].rename_column("label", "labels").remove_columns(["text"]),
            "imdb": imdb_datasets["train"].rename_column("label", "labels").remove_columns(["text"])}
test_ds = {"sst2": sst2_datasets["test"].rename_column("label", "labels").remove_columns(["sentence", "idx"]),
           "rotten_tomatoes": rotten_tomatoes_datasets["test"].rename_column("label", "labels").remove_columns(["text"]),
           "imdb": imdb_datasets["test"].rename_column("label", "labels").remove_columns(["text"])}

training_args = TrainingArguments("test_trainer",
                                  evaluation_strategy="epoch",
                                  logging_steps=50,
                                  num_train_epochs=1,
                                  remove_unused_columns=False,
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=4,
                                  gradient_checkpointing=True,
                                  fp16=True,
                                  half_precision_backend='auto',
                                  optim="adafactor")

data_collator = DataCollatorWithPadding(tokenizer)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules_mapping.get("gpt2"),
    lora_dropout=0.06,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

trainer = MLTrainer(model=model,
                    training_args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    lora_config=lora_config,
                    save_dir="sent",
                    compute_metrics=compute_metrics)




