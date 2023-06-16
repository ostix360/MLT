from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

import torch
from datasets import concatenate_datasets
from peft import LoraConfig, PeftModel
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
                 save_dir: str,
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
        if save_dir is None:
            raise ValueError("save_dir cannot be None")

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.loras = loras
        self.lora_config = lora_config
        self.save_dir = save_dir
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers

    def load_model(self, loras_to_add: list, train: bool = False):
        self.model.disable_adapter()
        for lora in loras_to_add:
            self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora, is_trainable=train)

    def load_MLModel(self):
        if not isinstance(self.model, PeftModel) and len(self.loras) == 0:  # create useless Lora?
            self.model: PeftModel = PeftModel(self.model, self.lora_config, list(self.train_dataset.keys())[0])  # get_peft_model(model, lora_config)
        elif len(self.loras) > 0:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, Path(f"{self.save_dir}/{self.loras[0]}"), self.loras[0])
            else:
                self.model.load_adapter(Path(f"{self.save_dir}/{self.loras[0]}"), self.loras[0])
            for lora in self.loras[1:]:
                self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora)

    def train(self):
        print("Starting training")
        previous_lora = []
        self.load_MLModel()
        for lora_name in self.train_dataset.keys():
            train_ds = self.train_dataset[lora_name]
            eval_ds = self.eval_dataset[lora_name]
            self.load_model(previous_lora)
            self.model.add_adapter(lora_name, self.lora_config)
            self.__train_lora(train_ds, eval_ds, lora_name=lora_name)
            self.model.save_pretrained(f"{self.save_dir}")  # TODO check peft config with multiple loras
            self.loras.append(lora_name)
            self.__train_loras(loras=previous_lora)
        print("Training finished")

    def __train_lora(self, train_ds, eval_ds, lora_name):
        print(f"Training {lora_name}")
        self.model.print_trainable_parameters()
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

    def __train_loras(self, loras):
        print("preparing datasets")
        train_ds = None
        eval_ds = None
        for lora_name in loras:
            if train_ds is None:
                train_ds = self.train_dataset[lora_name].copy()
                eval_ds = self.eval_dataset[lora_name].copy()
            else:
                train_ds = concatenate_datasets(train_ds, self.train_dataset[lora_name])
                eval_ds = concatenate_datasets(eval_ds, self.eval_dataset[lora_name])
        train_ds.shuffle()
        eval_ds.shuffle()
        self.load_model(loras, train=True)
        self.__train_lora(train_ds, eval_ds, " + ".join(loras))
        self.model.save_pretrained(f"{self.save_dir}")
        print("Training finished")
