import json
import math
import os
import random

from dataclasses import dataclass

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = None
MAX_ROWS = 0
MAX_TOKENS = 8

SEED = 42
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "INK-USC/numer_sense"
FALLBACK_TSV_URL = "https://raw.githubusercontent.com/INK-USC/NumerSense/main/data/validation.masked.tsv"

OUTPUT_JSON = "llama-numer_sense.json"

CANDIDATE_WORDS = [
	"zero",
	"one",
	"two",
	"three",
	"four",
	"five",
	"six",
	"seven",
	"eight",
	"nine",
	"ten",
]

LAMBDA_LIST = [-5.0, 0.0, 1.0, 5.0]

SYSTEM_PROMPT = "You are a helpful assistant."
USER_TEMPLATE = (
	"You are given a sentence where a number between 0 and 10 has been replaced by the token <mask>.\n"
	"Sentence: {sentence}\n\n"
	"Which single English word from [zero, one, two, three, four, five, six, seven, eight, nine, ten] "
	"best fills in for <mask>? Answer with just the word."
)


@dataclass
class ExampleResult:
	index: int
	x: str
	y: str
	target: str
	lmb: float
	hit1: int
	rank: int | None


def set_seed(seed: int) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def prepare_tokenizer_and_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
	token_kwargs = {}
	if HF_TOKEN:
		token_kwargs["token"] = HF_TOKEN
	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **token_kwargs)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		device_map="auto",
		torch_dtype=torch.bfloat16,
		**token_kwargs,
	)
	device = model.device
	model.eval()
	return tokenizer, model, device


def apply_chat_template(tokenizer: AutoTokenizer, sentence: str) -> str:
	content = USER_TEMPLATE.format(sentence=sentence)
	messages = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": content},
	]
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_lm_head_and_norms(model: AutoModelForCausalLM) -> tuple[torch.Tensor, torch.Tensor]:
	lm_head = model.get_output_embeddings()
	W = lm_head.weight.detach()
	assert W.dim() == 2
	with torch.no_grad():
		norms = torch.linalg.norm(W.float(), dim=1)
		inv_norms = torch.where(norms > 0, norms.reciprocal(), torch.zeros_like(norms))
		inv_norms = inv_norms.to(W.dtype)
	return W, inv_norms


def build_candidate_token_ids(tokenizer: AutoTokenizer) -> dict[str, int]:
	candidate_ids: dict[str, int] = {}
	for word in CANDIDATE_WORDS:
		ids_with_space = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
		if len(ids_with_space) == 1:
			candidate_ids[word] = ids_with_space[0]
			continue
		ids_plain = tokenizer(word, add_special_tokens=False)["input_ids"]
		if len(ids_plain) == 1:
			candidate_ids[word] = ids_plain[0]
			continue
		raise ValueError(f"Cannot map word {word!r} to a single token.")
	return candidate_ids


def first_step_scores(
	base_logits: torch.Tensor,
	W: torch.Tensor,
	inv_norms: torch.Tensor,
	lmb: float,
) -> torch.Tensor:
	assert base_logits.dim() == 1
	assert base_logits.shape[0] == W.shape[0]
	if base_logits.dtype != W.dtype:
		base_logits = base_logits.to(W.dtype)
	V, d = W.shape
	assert inv_norms.shape[0] == V
	probs = torch.softmax(base_logits.float(), dim=-1)
	weighted = probs * inv_norms.float()
	direction = weighted @ W.float()
	e_dot_d = (W.float() @ direction) * inv_norms.float()
	if lmb != 0.0:
		e_dot_d = e_dot_d.to(base_logits.dtype)
		adj = base_logits + (lmb * e_dot_d)
		return adj
	return base_logits

def load_data() -> list[dict]:
	try:
		ds = load_dataset(DATASET_ID, split="train", trust_remote_code=True)
	except Exception:
		ds = load_dataset("csv", data_files={"train": FALLBACK_TSV_URL}, sep="\t")["train"]
	if MAX_ROWS and MAX_ROWS > 0:
		n = min(MAX_ROWS, len(ds))
		ds = ds.select(range(n))

	rows: list[dict] = []
	candidates = set(CANDIDATE_WORDS)
	cols = list(ds.column_names)

	def try_named(item: dict) -> tuple[str | None, str | None]:
		sent = None
		for k in ["sentence", "probe", "text", "input"]:
			if k in item and isinstance(item[k], str):
				sent = item[k]
				break
		tgt = None
		for k in ["target", "answer", "label", "ground_truth", "gold"]:
			if k in item and isinstance(item[k], str):
				tgt = item[k]
				break
		return sent, tgt

	def autodetect_columns() -> tuple[str, str]:
		sent_col, tgt_col = None, None
		mask_counts = {}
		cand_counts = {}
		for c in cols:
			col_vals = ds[c]
			if all(isinstance(v, str) for v in col_vals):
				mask_counts[c] = sum(1 for v in col_vals if "<mask>" in v)
				cand_counts[c] = sum(1 for v in col_vals if v.strip().lower() in candidates)
		if mask_counts:
			sent_col = max(mask_counts, key=lambda k: mask_counts[k]) if max(mask_counts.values()) > 0 else None
		if cand_counts:
			tgt_col = max(cand_counts, key=lambda k: cand_counts[k]) if max(cand_counts.values()) > 0 else None
		if not sent_col or not tgt_col:
			raise ValueError("Could not locate target/sentence columns")
		return sent_col, tgt_col

	sent_col_name, tgt_col_name = None, None
	first = ds[0] if len(ds) > 0 else {}
	s_probe, t_probe = try_named(first)
	if s_probe and t_probe:
		sent_col_name = next(k for k in cols if k in first and first[k] == s_probe)
		tgt_col_name = next(k for k in cols if k in first and first[k] == t_probe)
	else:
		sent_col_name, tgt_col_name = autodetect_columns()

	for idx in range(len(ds)):
		item = ds[idx]
		sent = str(item[sent_col_name])
		tgt = str(item[tgt_col_name]).strip().lower()
		rows.append({"index": idx, "sentence": sent, "target": tgt})
	return rows


def evaluate_lambda(
	model: AutoModelForCausalLM,
	tokenizer: AutoTokenizer,
	device: torch.device,
	data_rows: list[dict],
	lmb: float,
	W: torch.Tensor,
	inv_norms: torch.Tensor,
	candidate_ids: dict[str, int],
) -> tuple[list[ExampleResult], float, float]:
	results: list[ExampleResult] = []
	hits = 0
	rr_sum = 0.0
	rr_count = 0
	total = len(data_rows)
	for row in tqdm(data_rows, desc=f"λ={lmb:+g}"):
		index = row["index"]
		sentence = row["sentence"]
		target = row["target"]
		prompt_text = apply_chat_template(tokenizer, sentence)
		tokenized = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
		input_ids = tokenized["input_ids"].to(device)
		attention_mask = tokenized.get("attention_mask")
		if attention_mask is not None:
			attention_mask = attention_mask.to(device)
		with torch.no_grad():
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		logits = outputs.logits
		assert logits.dim() == 3
		base_logits = logits[0, -1, :]
		assert base_logits.shape[0] == W.shape[0]
		adj = first_step_scores(base_logits, W, inv_norms, lmb)
		score_items: list[tuple[str, float]] = []
		for word, token_id in candidate_ids.items():
			score_items.append((word, float(adj[token_id].item())))
		score_items.sort(key=lambda p: p[1], reverse=True)
		predicted_word = score_items[0][0]
		hit = 1 if predicted_word == target else 0
		hits += hit
		rank = None
		if target in candidate_ids:
			order = [w for w, _ in score_items]
			if target in order:
				rank = 1 + order.index(target)
				rr_sum += 1.0 / rank
				rr_count += 1
		results.append(
			ExampleResult(
				index=index,
				x=prompt_text,
				y=predicted_word,
				target=target,
				lmb=lmb,
				hit1=hit,
				rank=rank,
			)
		)
	hit_at_1 = hits / max(1, total)
	if rr_count == 0:
		mrr = 0.0
		return results, hit_at_1, mrr
	mrr = rr_sum / rr_count
	return results, hit_at_1, mrr


def main() -> None:
	set_seed(SEED)
	tokenizer, model, device = prepare_tokenizer_and_model()
	W, inv_norms = get_lm_head_and_norms(model)
	W = W.to(device)
	inv_norms = inv_norms.to(device)
	candidate_ids = build_candidate_token_ids(tokenizer)
	data_rows = load_data()
	all_records: list[dict] = []
	metrics: dict[str, dict[str, str]] = {}
	for lmb in LAMBDA_LIST:
		run_results, hit_at_1, mrr = evaluate_lambda(
			model,
			tokenizer,
			device,
			data_rows,
			lmb,
			W,
			inv_norms,
			candidate_ids,
		)
		print(f"λ={lmb:+g} hit@1={hit_at_1:.8f} MRR={mrr:.8f}", flush=True)
		metrics[str(lmb)] = {
			"hit@1": f"{hit_at_1:.8f}",
			"MRR": f"{mrr:.8f}",
		}
		for r in run_results:
			all_records.append(
				{
					"index": r.index,
					"x": r.x,
					"y": r.y,
					"target": r.target,
					"lambda": r.lmb,
					"hit@1": r.hit1,
					"rank": r.rank,
				}
			)
	result_json = {
		"records": all_records,
		"metrics": metrics,
	}
	with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
		json.dump(result_json, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()
