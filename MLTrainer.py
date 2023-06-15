import peft
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel, set_peft_model_state_dict
from transformers import TrainingArguments, DataCollator, PreTrainedTokenizer, Trainer


class MLTrainer:
    def __init__(self, model,
                 train_dataset: dict,
                 eval_dataset: dict,
                 training_args: TrainingArguments,
                 data_collator: DataCollator,
                 tokenizer: PreTrainedTokenizer,
                 data_format: dict,
                 lora_config: LoraConfig,
                 loras=None):
        if loras is None:
            loras = []
        if not isinstance(data_format, dict):
            raise TypeError("data_format must be a dict")
        if not isinstance(loras, list):
            raise TypeError("loras must be a list")
        if not isinstance(training_args, TrainingArguments):
            raise TypeError("training_args must be a TrainingArguments object")
        if not isinstance(train_dataset, dict):
            raise TypeError("train_dataset must be a dict object")
        if not isinstance(eval_dataset, dict):
            raise TypeError("eval_dataset must be a dict object")
        if not isinstance(data_collator, DataCollator):
            raise TypeError("data_collator must be a DataCollator object")
        if not isinstance(tokenizer, PreTrainedTokenizer):
            raise TypeError("tokenizer must be a PreTrainedTokenizer object")
        if not isinstance(lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig object")
        if len(train_dataset) != len(eval_dataset):
            raise ValueError("train_dataset and eval_dataset must be of the same length")

        if not isinstance(model, PeftModel) and len(loras) == 0:
            model = get_peft_model(model, lora_config)
        elif len(loras) > 0:
            if not isinstance(model, PeftModel):
                model = PeftModel.from_pretrained(model, f"/{loras[0]}", loras[0])
            else:
                model.load_adapter(f"/{loras[0]}", loras[0])
            for lora in loras[1:]:
                model.load_adapter(f"/{lora}", lora)

        self.model: PeftModel = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.data_format = data_format
        self.loras = loras
        self.lora_config = lora_config

    def load_model(self, loras_to_add: list, train: bool = False):
        self.model.disable_adapter()
        for lora in loras_to_add:
            self.model.load_adapter(f"/{lora}", lora, is_trainable=train)

        pass

    def train(self):
        print("Starting training")
        previous_lora = []
        for key in self.train_dataset.keys():
            train = self.train_dataset[key]
            eval = self.eval_dataset[key]
            self.load_model(previous_lora)
            self.model.add_adapter(key, self.lora_config)
            self.__train_single_lora(train, eval)
        print("Training finished")
        pass

    def __train_single_lora(self, train, eval):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train,
            eval_dataset=eval,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.evaluate()
        self.model.save_pretrained() # TODO check peftconfig with multiple loras

    def __train_loras(self, loras, model, data):
        pass


