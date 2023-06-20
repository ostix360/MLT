import evaluate
import torch
from accelerate import Accelerator
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler


def datasets_post_process(tokenized_dataset):
    final_datasets = tokenized_dataset.remove_columns(["sentence1", "text", "idx"])
    final_datasets = final_datasets.rename_column("label", "labels")
    final_datasets.set_format(type="torch")
    # Debug
    print(final_datasets.column_names)
    return final_datasets


def debug_data_processing(train_dataloader):
    batch = None
    for batch in train_dataloader:
        break
    print({k: v.shape for k, v in batch.items()})
    return batch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_lr_scheduler(optimizer, train_dataloader, num_epochs):
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Debug
    print("Num training step: ", num_training_steps)
    return lr_scheduler, num_training_steps


def train_with_accelerator(model, train_dataloader, num_epochs, optimizer, lr_scheduler, accelerator, num_training_steps):
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def evaluate_with_accelerate(model, eval_dataloader: DataLoader, accelerator: Accelerator):
    metric = evaluate.load("glue", "mrpc")
    model.eval()

    eval_dataloader = accelerator.prepare(eval_dataloader)
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions), references=accelerator.gather(batch["labels"])
        )

    return metric.compute()


def train_model(model, train_dataloader, num_epochs, optimizer, lr_scheduler, device, num_training_steps):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def evaluate_model(model, eval_dataloader, device):
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()
