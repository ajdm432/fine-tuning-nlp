from tqdm import tqdm
import evaluate
import torch

METRIC = "rouge"
NUM_EXAMPLES = 3
MAX_SEQ_LENGTH = 2048
MAX_OUT_LENGTH = 150
DEFAULT_PROMPT = f"Summarize the following article in no more than {MAX_OUT_LENGTH} words.".strip()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

metric = evaluate.load(METRIC)

def promptify_single(article, prompt=DEFAULT_PROMPT):
    return f"<s>[INST]{prompt}\nArticle: {article}[\INST]"

def evaluate_model(model, tokenizer, data):
    example_input_output(model, tokenizer, data)
    # rouge_test(model, tokenizer, data)

def example_input_output(model, tokenizer, data):
    print("\nExample Input/Output...\n")
    for i in range(NUM_EXAMPLES):
        prompt = promptify_single(data["article"][i])
        print("INPUT:")
        print(prompt)

        tok = tokenizer(prompt, padding=True, return_tensors='pt', max_length=MAX_SEQ_LENGTH, truncation=True)["input_ids"]
        model_out = model.generate(tok,
                                   do_sample=True,
                                   temperature=0.7,
                                   top_p=0.95,
                                   top_k=40,
                                   max_new_tokens=MAX_OUT_LENGTH)
        new_tokens = model_out[0]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
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