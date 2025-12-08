import datetime
import json
import math
import os
import platform
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

HF_TOKEN = None
MAX_ROWS = 1000
MAX_TOKENS = 0

SEED = 42
random.seed(SEED)
set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]
LAMBDAS = [-5.0, 0.0, 1.0, 5.0]

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"
DATASET_SPLIT = "validation"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFERRED_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

OUT_DIR = Path("wikitext")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_SLICE = 32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def maybe_auth():
    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def get_tokenizer_and_model(model_id: str, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        token=HF_TOKEN,
    )
    if DEVICE != "cuda":
        model.to(DEVICE)
    model.eval()
    return tokenizer, model

def require(cond, msg):
    if not cond:
        raise ValueError(msg)

def check_shapes_and_dtypes(model):
    emb = model.get_input_embeddings().weight
    require(emb.ndim == 2, f"Embedding matrix must be rank-2, got {emb.shape}")
    vocab_size, hidden_dim = emb.shape
    require(emb.dtype in (torch.float16, torch.bfloat16, torch.float32), f"Unexpected embedding dtype {emb.dtype}")
    return vocab_size, hidden_dim, emb.dtype

def verify_dataset_structure(ds, split):
    require(split in ds, f"Missing split: {split}")
    feats = ds[split].features
    require("text" in feats and str(feats["text"].dtype) == "string", f"Expected 'text' as string, got: {feats}")

@torch.no_grad()
def make_vibe_matrix(model, device) -> torch.Tensor:
    W = model.get_input_embeddings().weight.detach()
    E = W.to(torch.float32)
    norms = E.norm(dim=1, keepdim=True).clamp_min(1e-12)
    E = E / norms
    return E.to(device, non_blocking=True)

@torch.no_grad()
def vibe_adjusted_logprobs(logits: torch.Tensor, E: torch.Tensor, lam: float) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    B, V = logits.shape
    require(V == E.shape[0], "Vocab mismatch between logits and E")
    logits = logits.to(torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    d = torch.matmul(E.T, probs.T)
    bias = torch.matmul(E, d).T
    scores = log_probs + lam * bias
    return scores - torch.logsumexp(scores, dim=-1, keepdim=True)

@torch.no_grad()
def stride_ppl_with_vibe(model, tokenizer, E: torch.Tensor, input_ids: torch.Tensor, lam: float, max_length: int, stride: int = 512) -> float:
    nll_sum = 0.0
    count = 0
    steps = list(range(0, input_ids.size(1), stride))
    for start in tqdm(steps, desc=f"PPL λ={lam:+g}", leave=False):
        end = min(start + max_length, input_ids.size(1))
        if end - start <= 1:
            break
        chunk = input_ids[:, start:end].to(DEVICE, non_blocking=True)
        out = model(input_ids=chunk)
        logits = out.logits[:, :-1, :]
        target = chunk[:, 1:]
        B, Lm1, V = logits.shape
        for t0 in range(0, Lm1, TIME_SLICE):
            t1 = min(t0 + TIME_SLICE, Lm1)
            logits_tb = logits[:, t0:t1, :].reshape(-1, V)
            target_tb = target[:, t0:t1].reshape(-1, 1)
            adj_log_probs_tb = vibe_adjusted_logprobs(logits_tb, E, lam)
            nll_tb = -adj_log_probs_tb.gather(dim=-1, index=target_tb).squeeze(-1)
            nll_sum += float(nll_tb.sum().cpu().item())
            count += int(nll_tb.numel())
    return math.exp(nll_sum / max(count, 1))

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_index>")
        sys.exit(1)
    
    try:
        model_idx = int(sys.argv[1])
        if not (0 <= model_idx < len(MODELS)):
            raise ValueError
    except ValueError:
        print(f"Invalid index. Please choose between 0 and {len(MODELS) - 1}")
        sys.exit(1)

    model_id = MODELS[model_idx]
    
    maybe_auth()
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, token=HF_TOKEN)
    verify_dataset_structure(ds, DATASET_SPLIT)
    val = ds[DATASET_SPLIT]
    
    if MAX_ROWS and MAX_ROWS > 0:
        val = val.select(range(min(MAX_ROWS, len(val))))
    
    texts = [ex["text"] for ex in val if isinstance(ex.get("text"), str)]
    raw_corpus = "\n".join(texts).strip()
    require(len(raw_corpus) > 0, "Empty dataset slice.")

    print(f"\n=== Loading: {model_id} | dtype={PREFERRED_DTYPE} ===")
    tokenizer, model = get_tokenizer_and_model(model_id, PREFERRED_DTYPE)
    V, d, emb_dtype = check_shapes_and_dtypes(model)
    print(f"Embedding matrix: V={V}, d={d}, dtype={emb_dtype}")
    
    E = make_vibe_matrix(model, DEVICE)
    
    with torch.no_grad():
        input_ids = tokenizer(raw_corpus, return_tensors="pt").input_ids
    
    require(input_ids.size(1) >= 2, "Tokenised length too short for perplexity.")
    max_length = getattr(model.config, "max_position_embeddings", 2048) or 2048
    
    model_dir = OUT_DIR / model_id.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_for_model = {}
    for lam in LAMBDAS:
        ppl = stride_ppl_with_vibe(
            model=model,
            tokenizer=tokenizer,
            E=E,
            input_ids=input_ids,
            lam=lam,
            max_length=max_length,
            stride=min(512, max_length),
        )
        metrics_for_model[f"{lam:+g}"] = float(f"{ppl:.8f}")
        print(f"[{model_id}] λ={lam:+g} | PPL={ppl:.8f}")
    
    save_json(metrics_for_model, model_dir / "results.json")
    
    details = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "device": DEVICE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "model_processed": model_id,
        "lambdas": LAMBDAS,
        "dtype_requested": str(PREFERRED_DTYPE),
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "max_rows": MAX_ROWS,
        }
    }
    save_json(details, model_dir / "setup.json")
    
    print(f"\n=== Done for {model_id}. Results in {model_dir.resolve()} ===\n")

if __name__ == "__main__":
    main()