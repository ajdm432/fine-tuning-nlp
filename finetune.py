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

MAX_SEQ_LENGTH = 4096

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

def promptify_data(examples, tokenizer):
    articles = examples['article']
    summaries = examples['highlights']
    texts = []
    for article, summary in zip(articles, summaries):
        text = f"### Article: {article}\n### Summary: {summary}" + tokenizer.eos_token
        texts.append(text)
    return {"text" : texts,}


def load_base_model_and_tokenizer():
    print(f"Loading Base Model {BASE_MODEL}...")
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
        # tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.pad_token_id = tokenizer.unk_token_id
        # tokenizer.padding_side = "right"
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_FILE,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            cache_dir=None,
        )
        # tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.pad_token_id = tokenizer.unk_token_id
        # tokenizer.padding_side = "right"
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

def train(model, tokenizer, train_dataset, val_dataset, checkpoint, checkpoint_name=None):
    # format dataset
    train_dataset = train_dataset.map(lambda x: promptify_data(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: promptify_data(x, tokenizer), batched=True)

    # peft params
    lora_alpha = 16
    lora_r = 16
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        random_state=43,
        use_rslora=False,
        loftq_config=None
    )

    # train params
    training_arguments = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        warmup_steps=5,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        do_eval=True,
        save_strategy="steps",
        save_steps=50,
        logging_steps=100,
        output_dir=OUTPUT_DIR,
        lr_scheduler_type="linear",
        seed=43,
        push_to_hub=False,
        save_total_limit=3,
        max_steps=3000,
    )

    # context_response_template = "\n### Summary:"
    # response_template_ids = tokenizer.encode(context_response_template, add_special_tokens=False)[2:]
    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

    print(f"Creating Trainer...")
    # create trainer object
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_arguments,
        packing=False,
        dataset_num_proc=2,
    )

    # begin training
    if checkpoint:
        print(f"Beginning Model Training From Checkpoint...")
        trainer.train(checkpoint_name)
    else:
        print(f"Beginning Model Training From Scratch...")
        trainer.train()
    print(f"Beginning Model Evaluation...")
    trainer.evaluate()
    # save tokenizer at same location as model for unsloth
    model.save_pretrained(TRAINED_MODEL_FILE, cache_dir=None)
    tokenizer.save_pretrained(TRAINED_MODEL_FILE, cache_dir=None)

if __name__=='__main__':
    # define argument options
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--train", dest="do_training", default=False, action="store_true")
    argparser.add_argument("-b", "--use-base", dest="use_base", default=False, action="store_true")
    argparser.add_argument("-c", "--use-checkpoint", dest="use_checkpoint", default=None, type=str)
    argparser.add_argument("-e", "--eval_checkpoint", dest="eval_checkpoint", default=False, action="store_true")
    opts = argparser.parse_args()

    if opts.use_checkpoint is not None:
        traindata, valdata, testdata = load_data()
        if not opts.eval_checkpoint:
            model, tokenizer = load_base_model_and_tokenizer()
            train(model, tokenizer, traindata, valdata, checkpoint=True, checkpoint_name=opts.use_checkpoint)
            model, tokenizer = load_trained_model_and_tokenizer()
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=opts.use_checkpoint,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
                cache_dir=None,
            )
            FastLanguageModel.for_inference(model)
            evaluate_model(model, tokenizer, testdata)
    else:
        if (not os.path.isdir(TRAINED_MODEL_FILE) or opts.do_training) and not opts.use_base:
            print("Could not find trained model file, begining training...")
            traindata, valdata, testdata = load_data()
            model, tokenizer = load_base_model_and_tokenizer()
            model.config.use_cache = False
            train(model, tokenizer, traindata, valdata, checkpoint=False)
        else:
            _, _, testdata = load_data()
            if opts.use_base:
                model, tokenizer = load_base_model_and_tokenizer()
            else:
                model, tokenizer = load_trained_model_and_tokenizer()
        FastLanguageModel.for_inference(model)
        evaluate_model(model, tokenizer, testdata)
