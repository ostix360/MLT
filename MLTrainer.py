from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

import torch
from datasets import concatenate_datasets, Dataset
from peft import LoraConfig, PeftModel, MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PromptLearningConfig
from peft.utils.other import _freeze_adapter, _get_submodules, ModulesToSaveWrapper
from peft.mapping import _prepare_prompt_learning_config
from torch.utils.data import DataLoader
from transformers import TrainingArguments, DataCollator, Trainer, EvalPrediction, \
    PreTrainedTokenizerBase


class MLTrainer:
    """
    Trainer class for Multiple Lora Training

    Example usage for train_datasets and eval_datasets (see below for more information):
        {"dataset_name": dataset, "dataset_name2": dataset2, ...}
    The dataset name is used to identify the dataset and the lora adapter in the training loop

    """

    def __init__(self, model,
                 train_datasets: dict,
                 eval_datasets: dict,
                 training_args: TrainingArguments,
                 data_collator: DataCollator,
                 tokenizer: PreTrainedTokenizerBase,
                 lora_config: LoraConfig,
                 train_ratio: float = 0.8,
                 finetune_first: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 loras=None):
        """
        :param transformers.PreTrainedModel model: The model to train
        :param dict train_datasets: a dict that contains the train datasets
        :param dict eval_datasets: a dict that contains the eval datasets
        :param TrainingArguments training_args: training arguments needed even for a custom training loop
        :param DataCollator data_collator: Needed for a custom or none custom training loop
        :param tokenizer: Used only for the finetuning and the none custom training loop
        :param LoraConfig lora_config: The general lora config for all Lora adapters
        :param int train_ratio: A Value between 0 and 1 that indicates the ratio of the train dataset when loras are trained together
        :param bool finetune_first: Set to True if you want to finetune the model first before training the loras
        :param compute_metrics: compute metrics function used in the transformers.Trainer class
        :param optimizers: optimizers used in the transformers.Trainer class
        :param list loras: A list of lora names that will be loaded locally
                    If you want to train them the name of the lora must be in the train_dataset and eval_dataset
                    Order of the loras is important because the order is used to identify the lora adapter

        The TrainingArguments, DataCollator and the compute_metrics function are used in the transformers.Trainer class
        See transformers documentation for more information
        """

        if loras is None:
            loras = []
        if not isinstance(loras, list):
            raise TypeError("loras must be a list")
        if not isinstance(training_args, TrainingArguments):
            raise TypeError("training_args must be a TrainingArguments object")
        if not isinstance(train_datasets, dict):
            raise TypeError("train_dataset must be a dict object")
        if not isinstance(eval_datasets, dict):
            raise TypeError("eval_dataset must be a dict object")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("tokenizer must be a PreTrainedTokenizerBase object")
        if not isinstance(lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig object")
        if len(train_datasets) != len(eval_datasets):
            raise ValueError("train_dataset and eval_dataset must be of the same length")

        self.model = model
        self.train_dataset = train_datasets
        self.eval_dataset = eval_datasets
        self.training_args = training_args
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.loras = loras
        self.finetune_first = finetune_first
        self.lora_config = lora_config
        self.save_dir = self.training_args.output_dir
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers
        self.train_ratio = train_ratio
        self.c_eval_dataset = None
        self.c_train_dataset = None

    def load_model(self, train: bool = False):
        """
        Load the adapters locally found in self.loras list
        And set them as trainable if train is True
        :param bool train: Set to True to set the adapters as trainable and False otherwise
        :return: Nothing
        """
        for lora in self.loras:
            self.model.load_adapter(Path(f"{self.save_dir}/{lora}"), lora, is_trainable=train)
            if train:
                unfreeze_adapter(self.model, lora)
                set_additional_trainable_modules(self.model, lora)
            else:
                _freeze_adapter(self.model, lora)

    def load_MLModel(self):
        """
        Convert the model to a PeftModel and load the adapters locally fround in self.loras list

        :return: True if an untrained lora has been added to the model, False otherwise
        """
        if len(self.loras) > 0:
            if not isinstance(self.model, PeftModel):
                self.model: PeftModel = PeftModel.from_pretrained(self.model, Path(f"{self.save_dir}/{self.loras[0]}"),
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
        return False

    def finetune(self):
        """
        Finetune the model with the first dataset using the transformer trainer
        And save the finetuned model
        :return:
        """
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
        self.model.name_or_path = f"{self.save_dir}"    # Change the model's name to the link with it model's adapters

    def train(self):
        """
        Train function for MLTrainer
        Starting by processing the datasets, then finetuning with the first dataset
        and save the finetuned model if needed
        And then create the Loras and train them with the transformer trainer
        Finally, save the model's adapters
        :return: Nothing
        """

        print("Starting training")
        j = 0
        previous_ds = []
        if self.finetune_first:
            self.finetune()
            previous_ds.append(list(self.train_dataset.keys())[0])
            j = 1
        lora_added = self.load_MLModel()

        for i in range(j, len(self.train_dataset)):
            ds_name = list(self.train_dataset.keys())[i]
            train_ds = self.train_dataset[ds_name]
            eval_ds = self.eval_dataset[ds_name]
            if not lora_added:
                self.model.add_adapter(ds_name, self.lora_config)
                set_additional_trainable_modules(self.model, ds_name)
                self.model.set_adapter(ds_name)

            self.train_lora(train_ds, eval_ds, lora_name=ds_name)
            self.model.save_pretrained(f"{self.save_dir}")
            if not lora_added:
                self.loras.append(ds_name)
            previous_ds.append(ds_name)
            self.train_loras(ds=previous_ds)
            lora_added = False
            self.load_model()

        print("Training finished")

    def custom_train(self, trainer):
        """
        Custom train function for MLTrainer
        Starting by processing the datasets, then finetuning with the first dataset
        and save the finetuned model if needed
        Be ware the finetuning method don't use the custom trainer
        And then create the Loras and train them with the custom trainer
        Finally, save the model's adapters
        :param func trainer: trainer is a function that takes a model, a train dataset and an eval dataset
                            and train the model
                            It's a custom training loop.
                            If null, the default transformer trainer is used
        :return: Nothing
        """
        print("Processing datasets")
        self.process_datasets()
        print("Starting training")
        j = 0
        previous_ds = []
        if self.finetune_first:
            self.finetune()
            j = 1
        print("Starting training")
        lora_added = self.load_MLModel()
        for i in range(j, len(self.train_dataset)):
            ds_name = list(self.train_dataset.keys())[i]
            train_ds = self.c_train_dataset[ds_name]
            eval_ds = self.c_eval_dataset[ds_name]
            if not lora_added:
                self.model.add_adapter(ds_name, self.lora_config)
                set_additional_trainable_modules(self.model, ds_name)
                self.model.set_adapter(ds_name)

            self.train_lora(train_ds, eval_ds, lora_name=ds_name, trainer=trainer)
            self.model.save_pretrained(f"{self.save_dir}")
            if not lora_added:
                self.loras.append(ds_name)
            previous_ds.append(ds_name)
            self.train_loras(ds=previous_ds, trainer=trainer)
            lora_added = False
            self.load_model()
        print("Training finished")

    def process_datasets(self):
        """
        Process the datasets to be used by the custom trainer
        :return: Nothing
        """
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
        """
        Train the model with the given dataset
        It can be one lora or all the loras depending on the training phase
        Be ware this function does not save the model
        :param train_ds: the dataset to use for training
        :param eval_ds: the dataset to use for evaluation
        :param lora_name: lora(s) to train
        :param func trainer: trainer is a function that takes a model, a train dataset and an eval dataset
                            and train the model
                            It's a custom training loop.
                            If null, the default transformer trainer is used
        :return: Nothing
        """
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

    def train_loras(self, ds: list, trainer=None):
        """
        Prepare the datasets and prepare the model by loading the adapters
        Give the dataset and the model to the trainer
        Finally save the model
        :param list ds: all the datasets to use for training that are in the self.train_dataset and self.eval_dataset
        :param func trainer: trainer is a function that takes a model, a train dataset and an eval dataset
                            and train the model
                            It's a custom training loop.
                            If null, the default transformer trainer is used
        :return: Nothing
        """
        print("preparing datasets")
        list_train_ds = []
        list_eval_ds = []
        for ds_name in ds:
            index_to_cut = int(len(self.train_dataset[ds_name]) * self.train_ratio)
            self.train_dataset[ds_name].shuffle()
            self.eval_dataset[ds_name].shuffle()
            list_train_ds.append(Dataset.from_dict(self.train_dataset[ds_name][:index_to_cut]))
            list_eval_ds.append(Dataset.from_dict(self.eval_dataset[ds_name][:index_to_cut]))

        train_ds = concatenate_datasets(list_train_ds)
        eval_ds = concatenate_datasets(list_eval_ds)
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
        model ([`transformers.PreTrainedModel`]): Model to be unfreeze.
        adapter_name ([str]): Name of the adapter to be unfreeze.
    """
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = True


def set_additional_trainable_modules(model, lora_name):
    """
    Sets additional trainable modules for a given adapter.
    Useful for classification models
        classifier layer is added as trainable module for the adapter
        Necessary for training to avoid a torch error
    :param model: Model
    :param str lora_name: the lora adapter name to add
    :return: Nothing
    """
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(lora_name)
            else:
                setattr(parent, target_name, ModulesToSaveWrapper(target, lora_name))
            for param in target.parameters():
                param.requires_grad = True
