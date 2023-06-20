import copy
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
from operator import itemgetter

import torch
from datasets import concatenate_datasets
from peft import LoraConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import TrainingArguments, DataCollator, PreTrainedTokenizer, Trainer, EvalPrediction, \
    PreTrainedTokenizerBase


class MLTrainer:
    def __init__(self, model,
                 train_dataset: dict,
                 eval_dataset: dict,
                 training_args: TrainingArguments,
                 data_collator: DataCollator,
                 tokenizer: PreTrainedTokenizerBase,
                 lora_config: LoraConfig,
                 finetune_first: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 loras=None):

        if loras is None:
            loras = []
        if not isinstance(loras, list):
            raise TypeError("loras must be a list")
        if not isinstance(training_args, TrainingArguments):
            raise TypeError("training_args must be a TrainingArguments object")
        if not isinstance(train_dataset, dict):
            raise TypeError("train_dataset must be a dict object")
        if not isinstance(eval_dataset, dict):
            raise TypeError("eval_dataset must be a dict object")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("tokenizer must be a PreTrainedTokenizerBase object")
        if not isinstance(lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig object")
        if len(train_dataset) != len(eval_dataset):
            raise ValueError("train_dataset and eval_dataset must be of the same length")

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.loras = loras
        self.finetune_first = finetune_first
        self.lora_config = lora_config
        self.save_dir = self.training_args.output_dir
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers
        self.c_eval_dataset = None
        self.c_train_dataset = None

    def load_model(self, train: bool = False):
        self.model.peft_config.clear()
        for lora in self.loras:
            self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora, is_trainable=train)

    def load_MLModel(self):
        if len(self.loras) > 0:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, Path(f"{self.save_dir}/{self.loras[0]}"),
                                                       self.loras[0])
            else:
                self.model.load_adapter(Path(f"{self.save_dir}/{self.loras[0]}"), self.loras[0])
            for lora in self.loras[1:]:
                self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora)
        elif not isinstance(self.model, PeftModel):  # create useless Lora?
            lora_name = list(self.train_dataset.keys())[1 if self.finetune_first else 0]
            self.model: PeftModel = PeftModel(self.model, self.lora_config, lora_name)
            self.loras.append(lora_name)
            return True
            # get_peft_model(model, lora_config)

        return False

    def finetune(self):     # TODO add removed dataset to train in train_loras
        print("Finetuning")
        train_ds = list(self.train_dataset.values())[0]
        eval_ds = list(self.eval_dataset.values())[0]
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=self.optimizers
        )
        trainer.train()
        trainer.evaluate()

    def train(self):
        print("Starting training")
        j = 0
        previous_ds = []
        if self.finetune_first:
            self.finetune()
            j = 1
        pefted = self.load_MLModel()
        if pefted:
            self.load_model(True)
        for i in range(j, len(self.train_dataset)):
            ds_name = list(self.train_dataset.keys())[i]
            train_ds = self.train_dataset[ds_name]
            eval_ds = self.eval_dataset[ds_name]
            if not pefted:
                self.model.add_adapter(ds_name, self.lora_config)
                self.model.set_adapter(ds_name)

            self.train_lora(train_ds, eval_ds, lora_name=ds_name)
            self.model.save_pretrained(f"{self.save_dir}")  # TODO check peft config with multiple loras
            if not pefted:
                self.loras.append(ds_name)
            previous_ds.append(ds_name)
            self.train_loras(ds=previous_ds)
            pefted = False
            self.load_model()

        print("Training finished")

    def custom_train(self, trainer):
        print("Processing datasets")
        self.process_datasets()
        print("Starting training")
        ds_lora = []
        self.load_MLModel()
        for lora_name in self.train_dataset.keys():
            train_ds = self.c_train_dataset[lora_name]
            eval_ds = self.c_eval_dataset[lora_name]
            self.load_model()
            self.model.add_adapter(lora_name, self.lora_config)
            self.train_lora(train_ds, eval_ds, lora_name=lora_name, trainer=trainer)
            self.model.save_pretrained(f"{self.save_dir}")  # TODO check peft config with multiple loras
            self.loras.append(lora_name)
            ds_lora.append(lora_name)
            self.train_loras(ds=ds_lora, trainer=trainer)
        print("Training finished")

    def process_datasets(self):
        self.c_train_dataset = {
            k: DataLoader(
                v.shuffle(),
                batch_size=self.training_args.per_device_train_batch_size,
                collate_fn=self.data_collator,
            )
            for k, v in self.train_dataset.items()
        }
        self.c_eval_dataset = {
            k: DataLoader(
                v.shuffle(),
                batch_size=self.training_args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
            )
            for k, v in self.eval_dataset.items()
        }

    def train_lora(self, train_ds, eval_ds, lora_name, trainer=None):
        print(f"Training {lora_name}")
        self.model.print_trainable_parameters()
        if trainer is None:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,

                compute_metrics=self.compute_metrics,
                optimizers=self.optimizers
            )
            trainer.train()
            trainer.evaluate()
        else:
            trainer(self.model, train_ds, eval_ds)

    def train_loras(self, ds, trainer=None):
        print("preparing datasets")
        train_ds = concatenate_datasets([self.train_dataset[ds_name] for ds_name in ds])   # TODO Take a part
        eval_ds = concatenate_datasets([self.eval_dataset[ds_name] for ds_name in ds])
        train_ds.shuffle()
        eval_ds.shuffle()
        if trainer is not None:
            train_ds = DataLoader(
                train_ds, batch_size=self.training_args.per_device_eval_batch_size, collate_fn=self.data_collator,
            )
            eval_ds = DataLoader(
                eval_ds, batch_size=self.training_args.per_device_eval_batch_size, collate_fn=self.data_collator,
            )
        self.load_model(train=True)
        self.train_lora(train_ds, eval_ds, " + ".join(ds), trainer=trainer)
        self.model.save_pretrained(f"{self.save_dir}")
