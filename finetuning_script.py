import os
import sys
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Optional, List, Callable, Union
from trl import DataCollatorForCompletionOnlyLM
from dataclasses import dataclass, field, InitVar
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, HfArgumentParser, TrainingArguments, Trainer



@dataclass
class ModelArguments:
    model_id: str                                = field(metadata={"help": "model name on HuggingFace or path to local model"})
    adapter_name: Optional[str]                  = field(default=None, metadata={"help": "adapter name on HuggingFace or path to local adapter"})
    load_in_4bit: InitVar[bool]                  = field(default=True, metadata={"help": "enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes"})
    bnb_4bit_compute_dtype: InitVar[torch.dtype] = field(default=torch.bfloat16, metadata={"help": "Computational type: Union[torch.bfloat16, torch.float16, torch.float32]"})
    bnb_4bit_quant_type: InitVar[str]            = field(default="nf4", metadata={"help": "quantization data type in the bnb.nn.Linear4Bit layers: Union['nf4', 'fp4']"})
    bnb_4bit_use_double_quant: InitVar[bool]     = field(default=False, metadata={"help": "enable nested quantization"})
    quant_config: BitsAndBytesConfig             = field(init=False)
    device_map: str                              = field(default="auto")
    output_hidden_states: InitVar[bool]          = field(default=False, metadata={"help": "outputs hidden states (W) during fwd pass"})
    output_attentions: InitVar[bool]             = field(default=False, metadata={"help": "outputs attentions calculated during fwd pass"})
    output_scores: InitVar[bool]                 = field(default=False, metadata={"help": "outputs logits calculated during fwd pass"})
    return_dict_in_generate: InitVar[bool]       = field(default=False, metadata={"help": "return ModelOutput during generation or a simple tuple"})
    config_args: dict                            = field(init=False)

    def __post_init__(self, load_in_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, bnb_4bit_use_double_quant,
                      output_hidden_states, output_attentions, output_scores, return_dict_in_generate):
        self.quant_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                               bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                                               bnb_4bit_quant_type=bnb_4bit_quant_type,
                                               bnb_4bit_use_double_quant=bnb_4bit_use_double_quant)

        self.config_args = {"output_hidden_states":output_hidden_states,
                            "output_attentions":output_attentions,
                            "output_scores": output_scores,
                            "return_dict_in_generate": return_dict_in_generate}

@dataclass
class PeftArguments:
    lora_r: int                                    = field(metadata={"help": "rank of the update matrices"})
    lora_alpha: int                                = field(metadata={"help": "alpha parameter for Lora scaling"})
    lora_dropout: float                            = field(metadata={"help": "dropout probability for Lora layers"})
    target_modules: List[str]                      = field(metadata={"help": "names of the modules to apply Lora to"})
    bias: str                                      = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation"})
    modules_to_save: Optional[List[str]]           = field(default=None, metadata={"help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task"})
    layers_to_transform: Optional[List[int]]       = field(default=None, metadata={"help": "List of layers to be transformed by LoRA. If not specified, all layers in target_modules are transformed"})

@dataclass
class DatasetArguments:
    dataset_id: str = field(metadata={"help": "dataset name on HuggingFace or path to Union[datasets.Dataset, csv, pandas.DataFrame, dict]"})
    valid_split_name: str = field(default="valid")
    train_split_name: str = field(default="train")
    validation_split_percentage: int = field(default=10)
    preprocessing_func: Optional[Callable] = field(default=None)
    num_workers: int = field(default=1)
    text_column_name: str = field(default="text")
    block_size: int = field(default=1024)
    postprocessing_func: Optional[Callable] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.validation_split_percentage > 100:
            self.validation_split_percentage = self.validation_split_percentage%100

@dataclass
class MyDefaultTrainingArguments(TrainingArguments):
    output_dir: str
    do_train: bool = field(default=True)
    remove_unused_columns: bool = False

@dataclass
class DataCollatorArguments:
    collator_type: str = field(default="Causal", metadata={"help": "One of 'Causal', 'Masked' or 'Supervised'"})
    response_template: str = field(default="### Response:\n")

    def __post_init__(self):
        if self.collator_type.lower() not in ["causal", "masked", "supervised"]:
            self.collator_type = "Causal"

@dataclass
class EvalArguments:
    metric_type: str = field(default="accuracy")

# PARSER
parser = HfArgumentParser((ModelArguments, PeftArguments, DatasetArguments, MyDefaultTrainingArguments, DataCollatorArguments, EvalArguments))
if len(sys.argv) == 2:
    if sys.argv[1].endswith(".json"):
        model_args, peft_args, dataset_args, training_arguments, dc_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif sys.argv[1].endswith(".yaml"):
        model_args, peft_args, dataset_args, training_arguments, dc_args, eval_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
else:
    model_args, peft_args, dataset_args, training_arguments, dc_args, eval_args = parser.parse_args_into_dataclasses()


print(model_args)
print(peft_args)
print(dataset_args)
print(training_arguments)
print(dc_args)
print(eval_args)

# Hacky fix for scientific notation in Yaml files
if isinstance(training_arguments.learning_rate, str):
    training_arguments.learning_rate = float(training_arguments.learning_rate)

import pdb; pdb.set_trace()

# LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(model_args.model_id,
                                             quantization_config=model_args.quant_config,
                                             device_map=model_args.device_map,
                                             **model_args.config_args)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_id, device_map=model_args.device_map)

# embedding size check
# idk why this is done (copied directly from HF script)
# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    print("Resizing embeddings to avoid index errors")
    model.resize_token_embeddings(len(tokenizer))



# PEFT
# peft_args = PeftArguments(lora_r=16, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj"])
config = LoraConfig(r=peft_args.lora_r,
                    lora_alpha=peft_args.lora_alpha,
                    lora_dropout=peft_args.lora_dropout,
                    bias=peft_args.bias,
                    target_modules=peft_args.target_modules,
                    modules_to_save=peft_args.modules_to_save,
                    layers_to_transform=peft_args.layers_to_transform,
                    task_type="CAUSAL_LM",
                    )
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)
model.gradient_checkpointing_enable()


# LOAD DATASET
# dataset_args = DatasetArguments("Abirate/english_quotes")
if dataset_args.dataset_id.endswith(".csv"):
    raw_datasets = Dataset.from_csv(dataset_args.dataset_id)
elif isinstance(dataset_args.dataset_id, dict):
    raw_datasets = Dataset.from_dict(dataset_args.dataset_id)
elif isinstance(dataset_args.dataset_id, pd.DataFrame):
    raw_datasets = Dataset.from_pandas(dataset_args.dataset_id)
else:
    raw_datasets = load_dataset(dataset_args.dataset_id)

if dataset_args.valid_split_name not in raw_datasets.keys():
    raw_datasets[dataset_args.valid_split_name] = load_dataset(dataset_args.dataset_id, split=f"train[:{dataset_args.validation_split_percentage}%]")
    raw_datasets[dataset_args.train_split_name] = load_dataset(dataset_args.dataset_id, split=f"train[{dataset_args.validation_split_percentage}%:]")

if callable(dataset_args.preprocessing_func):
    raw_datasets = raw_datasets.map(dataset_args.preprocessing_func, batched=True, num_proc=dataset_args.num_workers)

column_names = list(raw_datasets[dataset_args.train_split_name].features)
dataset_args.text_column_name = column_names[0] if dataset_args.text_column_name not in column_names else dataset_args.text_column_name
dataset_args.block_size = min(dataset_args.block_size, tokenizer.model_max_length)

def tokenize_function(examples):
    output = tokenizer(examples[dataset_args.text_column_name])
    return output

raw_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=dataset_args.num_workers, remove_columns=column_names)
raw_datasets = raw_datasets.filter(lambda example: len(example['input_ids']) < dataset_args.block_size)

if callable(dataset_args.postprocessing_func):
    raw_datasets = raw_datasets.map(dataset_args.postprocessing_func, batched=True, num_proc=dataset_args.num_workers)

train_dataset = raw_datasets[dataset_args.train_split_name]
if dataset_args.max_train_samples is not None:
    dataset_args.max_train_samples = min(len(train_dataset), dataset_args.max_train_samples)
    train_dataset = train_dataset.select(range(dataset_args.max_train_samples))

eval_dataset = raw_datasets[dataset_args.valid_split_name]
if dataset_args.max_eval_samples is not None:
    dataset_args.max_eval_samples = min(len(eval_dataset), dataset_args.max_eval_samples)
    eval_dataset = eval_dataset.select(range(dataset_args.max_eval_samples))




# DATA COLLATOR
if dc_args.collator_type.lower() == "masked":
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
elif dc_args.collator_type.lower() == "supervised":
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=dc_args.response_template)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TRAINER

import evaluate


if eval_args.metric_type.lower() == "perplexity":
    metric = evaluate.load("perplexity")
    metric_name = "perplexity"
else:
    metric = evaluate.load("accuracy")
    metric_name = "accuracy"


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    labels[labels == -100] = tokenizer.pad_token_id

    preds = preds[:, :-1].reshape(-1)
    preds[preds == -100] = tokenizer.pad_token_id

    if metric_name == "perplexity":
        return metric.compute(predictions=preds, references=labels) # for accuracy
    else:
        return metric.compute(predictions=tokenizer.decode(preds), model_id=model_args.model_id)


trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_arguments.do_eval else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_arguments.do_eval else None,
)


model.config.use_cache = False # silence the warnings. Please re-enable for inference!
trainer.train()
# trainer.train(resume_from_checkpoint=)
