import argparse, os
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from trl import SFTTrainer
from modeleval import evaluate_model

# get preferred device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set up training info
BASE_MODEL = "TheBloke/Mistral-7B-v0.1-GGUF/mistral-7b-v0.1.Q4_0.gguf"

# 3 splits, "train" (287k rows), "validation" (13.4k rows), and "test" (11.5k rows)
DATASET = "cnn_dailymail"
SUBSET = "2.0.0"

BASE_MODEL_FILE = "models/base_model"
TRAINED_MODEL_FILE = "models/out_model"
OUTPUT_DIR = "output"
DEFAULT_PROMPT = "Below is an article. Write a summary of the article.".strip()

DATA_PATH = "data/cnn_dailymail"

def load_data():
    if not os.path.isdir(DATA_PATH):
        print("Loading Dataset...")
        data = load_dataset(DATASET, name=SUBSET)
        # save dataset
        data.save_to_disk(DATA_PATH)
        # split up dataset
        train_data = data["train"]
        val_data = data["validation"]
        test_data = data["test"]
    else:
        data = load_from_disk(DATA_PATH)
        train_data = data["train"]
        val_data = data["validation"]
        test_data = data["test"]
    return train_data, val_data, test_data

def promptify_list(data):
    output_text = []
    for i in range(len(data["article"])):
        text = f"### Instruction: {DEFAULT_PROMPT} \n### Article: {data['article'][i]} \n### Summary: {data['highlights'][i]}"
        output_text.append(text)
    return output_text

def load_base_model_and_tokenizer():
    print(f"Loading Base Model {BASE_MODEL}...")
    if not os.path.isdir(BASE_MODEL_FILE):
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
        model.save_pretrained(BASE_MODEL_FILE)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_FILE,
            trust_remote_code=True,
        )

    print(f"Loading Tokenizer From Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def load_trained_model_and_tokenizer():
    print(f"Loading Saved Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_FILE,
        trust_remote_code=True,
    )
    print(f"Loading Model Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_FILE, trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def train(model, tokenizer, train_dataset, val_dataset):
    # peft params
    lora_alpha = 32
    lora_dropout = 0.05
    lora_r = 16
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # train params
    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        output_dir=OUTPUT_DIR,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    print(f"Creating Trainer...")
    # create trainer object
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
        formatting_func=promptify_list,
    )

    # begin training
    print(f"Beginning Model Training...")
    trainer.train()
    print(f"Beginning Model Evaluation...")
    trainer.evaluate()

if __name__=='__main__':
    # define argument options
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--train", dest="do_training", default=False, action="store_true")
    argparser.add_argument("-b", "--use-base", dest="use_base", default=False, action="store_true")
    opts = argparser.parse_args()

    if (not os.path.isdir(TRAINED_MODEL_FILE) or opts.do_training) and not opts.use_base:
        print("Could not find trained model file, begining training...")
        traindata, valdata, testdata = load_data()
        model, tokenizer = load_base_model_and_tokenizer()
        model.config.use_cache = False
        train(model, tokenizer, traindata, valdata)
    else:
        _, _, testdata = load_data()
        if opts.use_base:
            model, tokenizer = load_base_model_and_tokenizer()
        else:
            model, tokenizer = load_trained_model_and_tokenizer()
    evaluate_model(model, tokenizer, testdata)
