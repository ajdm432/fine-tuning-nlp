from tqdm import tqdm
import evaluate
import torch

METRIC = "rouge"
NUM_EXAMPLES = 3
MAX_SEQ_LENGTH = 4096
MAX_OUT_LENGTH = 20
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

metric = evaluate.load(METRIC)

def promptify_single(article):
    return f"###Article: {article}\n### Summary:"

def evaluate_model(model, tokenizer, data):
    example_input_output(model, tokenizer, data)
    rouge_test(model, tokenizer, data)

def example_input_output(model, tokenizer, data):
    print("\nExample Input/Output...\n")
    for i in range(NUM_EXAMPLES):
        prompt = promptify_single(data["article"][i])
        print("INPUT:")
        print(prompt)
        tokens = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
        tok_len = tokens["input_ids"].shape[1]
        model_out = model.generate(**tokens,
                                   do_sample=True,
                                   temperature=0.7,
                                   top_p=0.95,
                                   top_k=40,
                                   max_new_tokens=MAX_OUT_LENGTH,
                                   pad_token_id=tokenizer.eos_token_id)
        new_tokens = model_out[0, tok_len:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print("OUTPUT:")
        print(output)


def rouge_test(model, tokenizer, data):
    print("\nTesting Model...\n")
    outputs = []
    targets = data["highlights"]
    for model_in in tqdm(data["article"]):
        prompt = promptify_single(model_in)
        tokens = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
        tok_len = tokens["input_ids"].shape[1]
        model_out = model.generate(**tokens,
                                   do_sample=True,
                                   temperature=0.7,
                                   top_p=0.95,
                                   top_k=40,
                                   max_new_tokens=MAX_OUT_LENGTH,
                                   pad_token_id=tokenizer.eos_token_id)
        new_tokens = model_out[0, tok_len:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(output)

    results = metric.compute(predictions=outputs, references=targets)
    rouge1 = results['rouge1']
    rouge2 = results['rouge2']
    rougeL = results['rougeL']
    rougeLsum = results['rougeLsum']
    print(f"Rouge test results:\nrouge1:{rouge1}\nrouge2:{rouge2}\nrougeL:{rougeL}\nrougeLSum:{rougeLsum}")