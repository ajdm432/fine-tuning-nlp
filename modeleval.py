from tqdm import tqdm
import evaluate
import torch

METRIC = "rouge"
NUM_EXAMPLES = 3
MAX_SEQ_LENGTH = 2048
MAX_OUT_LENGTH = 100
DEFAULT_PROMPT = f"Summarize the following article in no more than {MAX_OUT_LENGTH} words.".strip()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

metric = evaluate.load(METRIC)

def promptify_single(article, prompt=DEFAULT_PROMPT):
    return f"[INST]{prompt}\nArticle: {article}[\INST]"

def evaluate_model(model, tokenizer, data):
    example_input_output(model, tokenizer, data)
    # rouge_test(model, tokenizer, data)

def example_input_output(model, tokenizer, data):
    print("\nExample Input/Output...\n")
    for i in range(NUM_EXAMPLES):
        prompt = promptify_single(data["article"][i])
        print("INPUT:")
        print(prompt)

        tok = tokenizer(prompt, padding=True, return_tensors='pt', max_length=MAX_SEQ_LENGTH, truncation=True)
        model_out = model.generate(input_ids=tok['input_ids'].to(DEVICE),
                                   attention_mask=tok['attention_mask'].to(DEVICE),
                                   max_length=MAX_SEQ_LENGTH,
                                   num_beams=5,
                                   early_stopping=True)
        new_tokens = model_out[0][tok['input_ids'].shape[-1]:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print("OUTPUT:")
        print(output)

    tok = tokenizer(prompt, padding=True, return_tensors='pt', max_length=MAX_SEQ_LENGTH, truncation=True)
    model_out = model.generate(input_ids=tok['input_ids'].to(DEVICE),
                                attention_mask=tok['attention_mask'].to(DEVICE),
                                max_length=MAX_SEQ_LENGTH,
                                num_beams=5,
                                early_stopping=True)
    output = tokenizer.decode(model_out[0], skip_special_tokens=True)
    print("OUTPUT:")
    print(output)


def rouge_test(model, tokenizer, data):
    print("\nTesting Model...\n")
    outputs = []
    targets = data["highlights"]
    for model_in in tqdm(data["article"]):
        prompt = promptify_single(model_in)
        tok = tokenizer.tokenize(prompt)
        model_out = model(tok)
        ids = tokenizer.convert_tokens_to_ids(model_out)
        output = tokenizer.decode(ids, skip_special_tokens=True)
        outputs.append(output)

    results = metric.compute(predictions=outputs, references=targets)
    rouge1 = results['rouge1']
    rouge2 = results['rouge2']
    rougeL = results['rougeL']
    rougeLsum = results['rougeLsum']
    print(f"Rouge test results:\nrouge1:{rouge1}\nrouge2:{rouge2}\nrougeL:{rougeL}\nrougeLSum{rougeLsum}")