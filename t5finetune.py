import datasets
import evaluate
import numpy as np
import pycountry
from datasets import load_dataset
from peft import LoraConfig, TaskType
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as TARGET_MODULES_MAPPING
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

from mlt.MLTrainer import MLTrainer

checkpoint = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, return_dict=True)

ds_name_list = ['de-en', 'de-fr', 'en-fr', ]
                # 'de-es', 'en-es',
                # 'de-fr', 'en-fr', 'es-fr',
                # 'de-hu', 'en-hu',          'fr-hu',
                # 'de-it', 'en-it', 'es-it', 'fr-it', 'hu-it',
                # 'de-nl', 'en-nl', 'es-nl', 'fr-nl', 'hu-nl',
                # 'de-ru', 'en-ru', 'es-ru', 'fr-ru', 'hu-ru', 'it-ru', ]


# ds_name_list = ['ca-de']


def load_all_subsets():
    ds_list = []
    for ds_name in ds_name_list:
        ds_list.append(load_dataset("opus_books", ds_name, split="train[1:]"))
    return ds_list


ds_list = load_all_subsets()

max_length = 512


def preprocess_function(examples):
    prefix1 = list(examples["translation"][0].keys())[0]
    prefix2 = list(examples["translation"][0].keys())[1]
    langs = prefix1 + "-" + prefix2
    lang1, lang2 = language_codes_to_names(langs)
    inputs = f"translate {lang1} to {lang2}: "
    inputs = [inputs + ex[prefix1] for ex in examples["translation"]]
    targets = [ex[prefix2] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def preprocess_function_swapped(examples):
    prefix1 = list(examples["translation"][0].keys())[1]
    prefix2 = list(examples["translation"][0].keys())[0]
    langs = prefix1 + "-" + prefix2
    lang1, lang2 = language_codes_to_names(langs)
    inputs = f"translate {lang1} to {lang2}: "
    inputs = [inputs + ex[prefix1] for ex in examples["translation"]]
    targets = [ex[prefix2] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


def language_codes_to_names(code: str):
    try:
        # Split the code into two parts
        lang1_code, lang2_code = code.lower().split('-')

        # Find the corresponding language names using pycountry
        lang1 = pycountry.languages.get(alpha_2=lang1_code)
        lang2 = pycountry.languages.get(alpha_2=lang2_code)
        if lang1.name == "Modern Greek (1453-)":
            lang1.name = "Modern Greek"
        return lang1.name, lang2.name

    except Exception as e:
        # Handle any errors during the conversion
        print(f"An error occurred: {e}")
        return None


tokenized_datasets = []

for ds in ds_list:
    tokenized_datasets.append(ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds.column_names,
    ))
    tokenized_datasets.append(ds.map(
        preprocess_function_swapped,
        batched=True,
        remove_columns=ds.column_names,
    ))

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, max_length=max_length)
metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    print(result)
    return {"bleu": result["score"]}


args = Seq2SeqTrainingArguments(
    f"t5-finetuned",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=12,
    save_total_limit=2,
    num_train_epochs=1,
    logging_steps=1500,
    predict_with_generate=True,
    fp16=True,
    optim="adafactor",
    push_to_hub=False,
)


def train(model, train_ds, eval_ds, ):
    local_trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    local_trainer.evaluate(max_length=max_length)
    local_trainer.train()
    local_trainer.evaluate(max_length=max_length)


lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=TARGET_MODULES_MAPPING["t5"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

train_ds = {}
test_ds = {}
i = 0

for ds_name in ds_name_list:
    lang1, lang2 = ds_name.split("-")
    t_ds = tokenized_datasets[i]
    train_ds[ds_name] = t_ds
    e_ds = t_ds[:200]
    test_ds[ds_name] = datasets.Dataset.from_dict(e_ds)
    i += 1
    ds_name = lang2 + "-" + lang1
    t_ds = tokenized_datasets[i]
    train_ds[ds_name] = t_ds
    e_ds = t_ds[:200]
    test_ds[ds_name] = datasets.Dataset.from_dict(e_ds)
    i += 1


trainer = MLTrainer(model=model,
                    finetune_first=False,
                    training_args=args,
                    train_datasets=train_ds,
                    eval_datasets=test_ds,
                    data_collator=data_collator,
                    lora_config=lora_config,
                    loras=["de-en", "en-de"],
                    tokenizer=tokenizer,
                    train_ratio=0.3,  # 30% of the dataset will be used for the multiple lora training part
                    )

trainer.train(train)
