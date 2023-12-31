import evaluate
import torch
from accelerate import Accelerator
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
import time


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


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


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
    t_batch = []
    t_forward = []
    t_backward = []
    t_step = []
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            start = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            t_batch.append(time.time() - start)

            start = time.time()
            output = model(**batch)
            t_forward.append(time.time() - start)

            start = time.time()
            loss = output.loss
            loss.backward()
            t_backward.append(time.time() - start)

            start = time.time()
            optimizer.step()
            lr_scheduler.step()
            t_step.append(time.time() - start)
            optimizer.zero_grad()
            progress_bar.update(1)

    print("Average time for batch: ", sum(t_batch) / len(t_batch))
    print("Average time for forward: ", sum(t_forward) / len(t_forward))
    print("Average time for backward: ", sum(t_backward) / len(t_backward))
    print("Average time for step: ", sum(t_step) / len(t_step))


def evaluate_model(model, eval_dataloader, device):
    metric = evaluate.load("accuracy")
    progress_bar = tqdm(range(eval_dataloader.__len__()))
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)

    return metric.compute()
