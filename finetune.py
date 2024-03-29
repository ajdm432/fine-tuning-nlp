import argparse, os
import torch
from datasets import load_dataset, load_from_disk, disable_caching
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from modeleval import evaluate_model

os.environ['HF_HOME'] = '.'
disable_caching()

# get preferred device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set up training info
BASE_MODEL = "unsloth/mistral-7b-bnb-4bit"

# 3 splits, "train" (287k rows), "validation" (13.4k rows), and "test" (11.5k rows)
DATASET = "cnn_dailymail"
SUBSET = "2.0.0"

BASE_MODEL_FILE = "models/base_model"
TRAINED_MODEL_FILE = "models/out_model"
OUTPUT_DIR = "output"
DEFAULT_PROMPT = "Below is an article. Write a summary of the article.".strip()

DATA_PATH = "data/cnn_dailymail"

MAX_SEQ_LENGTH = 2048

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

def promptify_data(examples):
    # inputs = [f"Article: {article}\nSummary:" for article in examples["article"]]
    # model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True)
    # labels = tokenizer(examples["highlights"], max_length=MAX_SEQ_LENGTH, truncation=True)
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs
    return [f"### Article: {examples['article'][i]}\n### Summary: {examples['highlights'][i]}" for i in range(len(examples["article"]))]

def load_base_model_and_tokenizer():
    print(f"Loading Base Model {BASE_MODEL[0]}...")
    if not os.path.isdir(BASE_MODEL_FILE):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            cache_dir=None,
        )
        model.save_pretrained(BASE_MODEL_FILE, cache_dir=None)
        tokenizer.save_pretrained(BASE_MODEL_FILE, cache_dir=None)
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "right"
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_FILE,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            cache_dir=None,
        )
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "right"
    return model, tokenizer

def load_trained_model_and_tokenizer():
    print(f"Loading Saved Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        TRAINED_MODEL_FILE,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        cache_dir=None,
    )
    return model, tokenizer

def train(model, tokenizer, train_dataset, val_dataset):
    # format dataset
    # train_dataset = train_dataset.map(lambda x: promptify_data(x, tokenizer), batched=True)
    # val_dataset = val_dataset.map(lambda x: promptify_data(x, tokenizer), batched=True)

    # peft params
    lora_alpha = 32
    lora_r = 32
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        use_gradient_checkpointing=False,
        random_state=43,
        use_rslora=False,
        loftq_config=None
    )

    # train params
    training_arguments = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        logging_steps=1,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        eval_steps=0.2,
        weight_decay=0.1,
        save_strategy="epoch",
        group_by_length=True,
        output_dir=OUTPUT_DIR,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=43,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    context_response_template = "\n### Summary:"
    response_template_ids = tokenizer.encode(context_response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

    print(f"Creating Trainer...")
    # create trainer object
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_arguments,
        packing=False,
        formatting_func=promptify_data,
        data_collator=collator,
    )

    # begin training
    print(f"Beginning Model Training...")
    trainer.train()
    print(f"Beginning Model Evaluation...")
    trainer.evaluate()
    # save tokenizer at same location as model for unsloth
    tokenizer.save_pretrained(TRAINED_MODEL_FILE, cache_dir=None)

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
    FastLanguageModel.for_inference(model)
    evaluate_model(model, tokenizer, testdata)
