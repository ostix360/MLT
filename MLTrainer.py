from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

import torch
from datasets import concatenate_datasets
from peft import LoraConfig, PeftModel, prepare_model_for_int8_training, \
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PromptLearningConfig
from peft.utils.other import _freeze_adapter, _set_trainable, _get_submodules, ModulesToSaveWrapper
from peft.mapping import _prepare_prompt_learning_config
from torch.utils.data import DataLoader
from transformers import TrainingArguments, DataCollator, Trainer, EvalPrediction, \
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
        for lora in self.loras:
            self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora, is_trainable=train)
            if train:
                unfreeze_adapter(self.model, lora)
                set_additional_trainable_modules(self.model, lora)
            else:
                _freeze_adapter(self.model, lora)

    def load_MLModel(self):
        if len(self.loras) > 0:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, Path(f"{self.save_dir}/{self.loras[0]}"),
                                                       adapter_name=self.loras[0])

            else:
                self.model.load_adapter(Path(f"{self.save_dir}/{self.loras[0]}"), self.loras[0])
            set_additional_trainable_modules(self.model, self.loras[0])
            for lora in self.loras[1:]:
                self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora)
                set_additional_trainable_modules(self.model, lora)
        elif not isinstance(self.model, PeftModel):  # create useless Lora?
            lora_name = list(self.train_dataset.keys())[1 if self.finetune_first else 0]
            self.model: PeftModel = get_peft_model(self.model, self.lora_config, lora_name)
            self.loras.append(lora_name)
            return True
            # get_peft_model(model, lora_config)

        return False

    def finetune(self):  # TODO add removed dataset to train in train_loras
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
        trainer.save_model(f"{self.save_dir}")
        self.model.name_or_path = f"{self.save_dir}"

    def train(self):
        print("Starting training")
        j = 0
        previous_ds = []
        if self.finetune_first:
            self.finetune()
            j = 1
        pefted = self.load_MLModel()

        for i in range(j, len(self.train_dataset)):
            ds_name = list(self.train_dataset.keys())[i]
            train_ds = self.train_dataset[ds_name]
            eval_ds = self.eval_dataset[ds_name]
            if not pefted:
                self.model.add_adapter(ds_name, self.lora_config)
                set_additional_trainable_modules(self.model, ds_name)
                self.model.set_adapter(ds_name)

            self.train_lora(train_ds, eval_ds, lora_name=ds_name)
            self.model.save_pretrained(f"{self.save_dir}")
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
        j = 0
        previous_ds = []
        if self.finetune_first:
            self.finetune()
            j = 1
        print("Starting training")
        pefted = self.load_MLModel()
        for i in range(j, len(self.train_dataset)):
            ds_name = list(self.train_dataset.keys())[i]
            train_ds = self.c_train_dataset[ds_name]
            eval_ds = self.c_eval_dataset[ds_name]
            if not pefted:
                self.model.add_adapter(ds_name, self.lora_config)
                set_additional_trainable_modules(self.model, ds_name)
                self.model.set_adapter(ds_name)

            self.train_lora(train_ds, eval_ds, lora_name=ds_name, trainer=trainer)
            self.model.save_pretrained(f"{self.save_dir}")
            if not pefted:
                self.loras.append(ds_name)
            previous_ds.append(ds_name)
            self.train_loras(ds=previous_ds, trainer=trainer)
            pefted = False
            self.load_model()
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
        train_ds = concatenate_datasets([self.train_dataset[ds_name] for ds_name in ds])  # TODO Take a part
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


def get_peft_model(model, peft_config, adapter_name):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
        adapter_name ([str]): Name of the adapter to be used.
    """
    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not isinstance(
            peft_config, PromptLearningConfig
    ):
        return PeftModel(model, peft_config)
    if isinstance(peft_config, PromptLearningConfig):
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)


def unfreeze_adapter(model, adapter_name):
    """
    Unfreezes an adapter.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be unfreezed.
        adapter_name ([str]): Name of the adapter to be unfreezed.
    """
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = True


def set_additional_trainable_modules(model, lora_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(lora_name)

            for param in target.parameters():
                param.requires_grad = True
            setattr(parent, target_name, ModulesToSaveWrapper(target, lora_name))

