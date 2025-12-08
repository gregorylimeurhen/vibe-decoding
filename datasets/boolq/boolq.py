import os
import sys
import json
import time
import random
import numpy
import torch
from datasets import load_dataset
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_ROWS = 0
HF_TOKEN = None

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct"
]

def set_seed(seed_value):
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def build_vibe_embeddings(model, device):
    embedding_module = model.get_input_embeddings()
    embedding_weight = embedding_module.weight.detach().float().to(device)
    norms = embedding_weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return embedding_weight / norms

def compute_label_base_and_alignment(prefix_ids, label_token_ids, model, vibe_embeddings, device):
    total_log_prob = 0.0
    total_alignment = 0.0
    prefix = list(prefix_ids)
    
    for token_id in label_token_ids:
        input_tensor = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs.logits[:, -1, :].float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            
            d_x = torch.matmul(probs, vibe_embeddings)
            e_token = vibe_embeddings[token_id]
            alignment = torch.dot(e_token, d_x.squeeze(0))
            log_prob_token = log_probs[0, token_id]
            
        total_log_prob += float(log_prob_token.detach().cpu())
        total_alignment += float(alignment.detach().cpu())
        prefix.append(int(token_id))
        
    return total_log_prob, total_alignment

def compute_metrics(true_labels, predictions_by_alpha):
    labels_tensor = torch.tensor(true_labels, dtype=torch.long)
    metrics = {}
    
    for alpha_key, preds in predictions_by_alpha.items():
        preds_tensor = torch.tensor(preds, dtype=torch.long)
        correct = int((preds_tensor == labels_tensor).sum().item())
        accuracy_value = correct / len(true_labels) if len(true_labels) > 0 else 0.0
        
        class_values = [0, 1]
        f1_macro = 0.0
        
        for class_value in class_values:
            tp = int(((labels_tensor == class_value) & (preds_tensor == class_value)).sum().item())
            fp = int(((labels_tensor != class_value) & (preds_tensor == class_value)).sum().item())
            fn = int(((labels_tensor == class_value) & (preds_tensor != class_value)).sum().item())
            
            if tp == 0 and fp == 0 and fn == 0:
                f1_c = 0.0
            else:
                precision_c = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall_c = tp / (tp + fn) if tp + fn > 0 else 0.0
                if precision_c + recall_c == 0.0:
                    f1_c = 0.0
                else:
                    f1_c = 2.0 * precision_c * recall_c / (precision_c + recall_c)
            f1_macro += f1_c
            
        f1_macro /= len(class_values)
        metrics[alpha_key] = {
            "accuracy": float(accuracy_value),
            "macro_f1": float(f1_macro),
        }
    return metrics

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_index>")
        sys.exit(1)
        
    try:
        model_idx = int(sys.argv[1])
        if model_idx not in [0, 1, 2]:
            raise ValueError
    except ValueError:
        print("Error: Index must be 0, 1, or 2.")
        sys.exit(1)

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_dataset("google/boolq")
    val_dataset = dataset["validation"]
    total_rows = len(val_dataset)
    
    if MAX_ROWS and MAX_ROWS > 0:
        total_rows = min(total_rows, int(MAX_ROWS))
        
    model_name = MODELS[model_idx]
    safe_model_name = model_name.replace("/", "__")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        )
    model.to(device)
    model.eval()
    
    vibe_embeddings = build_vibe_embeddings(model, device)
    
    label_texts_by_name = {
        "yes": " yes",
        "no": " no",
    }
    label_token_ids = {}
    for label_name, label_text in label_texts_by_name.items():
        encoded = tokenizer(label_text, add_special_tokens=False)
        label_token_ids[label_name] = [int(t) for t in encoded["input_ids"]]
        
    alphas = [-5.0, 0.0, 1.0, 5.0]
    alpha_keys = [str(a) for a in alphas]
    predictions_by_alpha = {alpha_key: [] for alpha_key in alpha_keys}
    true_labels = []
    examples_records = []
    
    start_time = time.time()
    
    progress_columns = [
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    
    with Progress(*progress_columns) as progress:
        task_id = progress.add_task(f"Evaluating {model_name}", total=total_rows)
        
        for idx in range(total_rows):
            row = val_dataset[int(idx)]
            question = str(row["question"])
            passage = str(row["passage"])
            answer_bool = bool(row["answer"])
            
            gold_label_name = "yes" if answer_bool else "no"
            gold_label_id = 1 if answer_bool else 0
            true_labels.append(gold_label_id)
            
            prompt_text = (
                "Passage:\n" + passage + "\n\nQuestion:\n" + question + 
                "\n\nAnswer the question with a single word: yes or no."
            )
            
            messages = [
                {"role": "system", "content": 'You are a question answering assistant. Answer with a single word: "yes" or "no".'},
                {"role": "user", "content": prompt_text},
            ]
            
            try:
                base_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                if "System role not supported" in str(e):
                    messages = [
                        {"role": "user", "content": 'You are a question answering assistant. Answer with a single word: "yes" or "no".\n\n' + prompt_text},
                    ]
                    base_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    raise e

            encoded_input = tokenizer(base_text, return_tensors="pt")
            prefix_ids = encoded_input["input_ids"][0].tolist()
            
            base_scores = {}
            alignments = {}
            
            for label_name in label_texts_by_name.keys():
                label_ids = label_token_ids[label_name]
                base_log_prob, total_alignment = compute_label_base_and_alignment(
                    prefix_ids, label_ids, model, vibe_embeddings, device
                )
                base_scores[label_name] = float(base_log_prob)
                alignments[label_name] = float(total_alignment)
                
            predictions_for_example = {}
            for alpha in alphas:
                alpha_key = str(alpha)
                label_scores = {}
                for label_name in label_texts_by_name.keys():
                    score_value = base_scores[label_name] + alpha * alignments[label_name]
                    label_scores[label_name] = float(score_value)
                    
                if label_scores["yes"] >= label_scores["no"]:
                    pred_label_name = "yes"
                else:
                    pred_label_name = "no"
                    
                pred_label_id = 1 if pred_label_name == "yes" else 0
                predictions_by_alpha[alpha_key].append(pred_label_id)
                predictions_for_example[alpha_key] = {
                    "predicted_label": pred_label_name,
                    "scores": {
                        "yes": float(label_scores["yes"]),
                        "no": float(label_scores["no"]),
                    },
                }
                
            example_record = {
                "index": int(idx),
                "question": question,
                "passage": passage,
                "gold_label": gold_label_name,
                "input_messages": messages,
                "input_text": base_text,
                "label_token_ids": {k: [int(t) for t in v] for k, v in label_token_ids.items()},
                "predictions": predictions_for_example,
            }
            examples_records.append(example_record)
            progress.advance(task_id, 1)
            
    end_time = time.time()
    metrics = compute_metrics(true_labels, predictions_by_alpha)
    
    setup_data = {
        "model_name": model_name,
        "dataset_name": "google/boolq",
        "split": "validation",
        "alphas": [float(a) for a in alphas],
        "max_rows": int(MAX_ROWS),
        "num_examples_evaluated": int(len(true_labels)),
        "random_seed": 42,
        "device": str(device),
        "vibe_method": "Label scoring with sequential vibe-adjusted token log-probabilities for labels 'yes' and 'no'.",
        "label_texts_by_name": label_texts_by_name,
        "library_versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch": str(torch.__version__),
            "transformers": str(torch.__dict__.get("__package__", "transformers")),
        },
        "runtime_seconds": float(end_time - start_time),
        "remarks": f"{model_name} evaluated on BoolQ using vibe decoding over label strings.",
    }
    
    results_data = {
        "model_name": model_name,
        "dataset_name": "google/boolq",
        "split": "validation",
        "alphas": [float(a) for a in alphas],
        "metrics": metrics,
        "examples": examples_records,
    }
    
    with open(f"setup_{safe_model_name}.json", "w", encoding="utf-8") as f_setup:
        json.dump(setup_data, f_setup, ensure_ascii=False, indent=2)
        
    with open(f"results_{safe_model_name}.json", "w", encoding="utf-8") as f_results:
        json.dump(results_data, f_results, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()