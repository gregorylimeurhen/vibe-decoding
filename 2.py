#= dataset: google/synthetic-persona-chat
#= model: meta-llama/llama-3.2-1b-instruct

import evaluate, os, re, torch, yaml
import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	LogitsProcessor,
	LogitsProcessorList,
	set_seed,
)

MAX_ROWS = 0

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_ID = "google/Synthetic-Persona-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

set_seed(SEED)

print(f"Running on: {DEVICE}")
print(f"Max rows: {MAX_ROWS if MAX_ROWS > 0 else 'Full Dataset'}")


class VibeLogitsProcessor(LogitsProcessor):
	def __init__(self, embeddings_matrix):
		with torch.no_grad():
			embeddings_float = embeddings_matrix.detach().float()
			norms = torch.norm(embeddings_float, p=2, dim=1, keepdim=True)
			norms[norms == 0] = 1e-9
			self.E = embeddings_float / norms

	def __call__(self, input_ids, scores):
		E = self.E.to(device=scores.device)
		probs = torch.softmax(scores.float(), dim=-1)

		# Calculate the "direction" of the current probability distribution
		direction_batch = torch.matmul(probs, E)

		# Project that direction back onto the vocabulary
		alignment_term = torch.matmul(direction_batch, E.T)

		log_probs = torch.log_softmax(scores.float(), dim=-1)
		final_scores = log_probs + alignment_term
		return final_scores.to(dtype=scores.dtype)


print(f"Loading model: {MODEL_ID}...")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

# Fix for Llama models which often lack a default pad token
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token
	print("Tokenizer pad_token was None. Set to eos_token.")

# Load Model
model = AutoModelForCausalLM.from_pretrained(
	MODEL_ID,
	torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
	device_map="auto",
	token=HF_TOKEN,
)

# Retrieve Embeddings for Vibe Processor
embedding_layer = model.get_input_embeddings()
vibe_processor = VibeLogitsProcessor(embedding_layer.weight)

print(f"Loading dataset: {DATASET_ID}...")
ds = load_dataset(DATASET_ID, split="train")

if MAX_ROWS > 0:
	ds = ds.select(range(MAX_ROWS))

results = []

print("Starting Generation...")

for row in tqdm(ds, desc="Evaluated Samples"):
	text_full = row["Best Generated Conversation"]

	# Parsing logic specific to google/synthetic-persona-chat
	match = re.search(r"User 1:|User 2:", text_full)
	if not match:
		continue

	text_conv = text_full[match.start():]
	parts = re.split(r"(User 1:|User 2:)", text_conv)
	if len(parts) < 3:
		continue

	turns = []
	for speaker, utterance in zip(parts[1::2], parts[2::2]):
		content = utterance.strip()
		if not content:
			continue
		role = "user" if "User 1" in speaker else "assistant"
		turns.append((role, content))

	if len(turns) < 2:
		continue
	if turns[-1][0] != "assistant":
		continue

	input_turns = turns[:-1]
	if input_turns[-1][0] != "user":
		continue

	target_response = turns[-1][1]
	messages = [{"role": role, "content": content} for role, content in input_turns]

	text_input = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)

	inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)

	greedy_out = model.generate(
		**inputs,
		max_new_tokens=64,
		do_sample=False,
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)
	greedy_text = tokenizer.decode(
		greedy_out[0][inputs.input_ids.shape[1]:],
		skip_special_tokens=True,
	)

	# 2. Vibe Generation
	vibe_out = model.generate(
		**inputs,
		max_new_tokens=64,
		do_sample=False,
		pad_token_id=tokenizer.pad_token_id,
		eos_token_id=tokenizer.eos_token_id,
		logits_processor=LogitsProcessorList([vibe_processor]),
	)
	vibe_text = tokenizer.decode(
		vibe_out[0][inputs.input_ids.shape[1]:],
		skip_special_tokens=True,
	)

	results.append(
		{
			"input": text_input,
			"reference": target_response,
			"greedy_prediction": greedy_text,
			"vibe_prediction": vibe_text,
		}
	)

print(f"Generated {len(results)} samples")

if len(results) == 0:
	raise ValueError("No usable samples generated. Check dataset parsing or MAX_ROWS.")

print("Calculating Metrics...")

metrics = {
	"bleu": evaluate.load("bleu"),
	"rouge": evaluate.load("rouge"),
	"bertscore": evaluate.load("bertscore"),
	"chrf": evaluate.load("chrf"),
	"sacrebleu": evaluate.load("sacrebleu"),
}

references = [sample["reference"] for sample in results]
greedy_predictions = [sample["greedy_prediction"] for sample in results]
vibe_predictions = [sample["vibe_prediction"] for sample in results]


def compute_all_metrics(predictions, references_local, name):
	scores = {}

	print(f"Computing {name} BLEU...")
	scores["bleu"] = metrics["bleu"].compute(
		predictions=predictions,
		references=[[ref] for ref in references_local],
	)["bleu"]

	print(f"Computing {name} SacreBLEU...")
	scores["sacrebleu"] = metrics["sacrebleu"].compute(
		predictions=predictions,
		references=[[ref] for ref in references_local],
	)["score"]

	print(f"Computing {name} ROUGE...")
	rouge_scores = metrics["rouge"].compute(
		predictions=predictions,
		references=references_local,
	)
	scores["rouge1"] = rouge_scores["rouge1"]
	scores["rougeL"] = rouge_scores["rougeL"]

	print(f"Computing {name} ChrF...")
	scores["chrf"] = metrics["chrf"].compute(
		predictions=predictions,
		references=references_local,
	)["score"]

	print(f"Computing {name} BERTScore...")
	bert_scores = metrics["bertscore"].compute(
		predictions=predictions,
		references=references_local,
		lang="en",
		device=DEVICE,
		model_type="bert-base-uncased", # Explicitly set for speed/consistency
	)
	scores["bertscore_f1"] = float(np.mean(bert_scores["f1"]))

	return scores


greedy_scores = compute_all_metrics(greedy_predictions, references, "Greedy")
vibe_scores = compute_all_metrics(vibe_predictions, references, "Vibe")

output_data = {
	"config": {
		"model": MODEL_ID,
		"dataset": DATASET_ID,
		"max_rows": MAX_ROWS,
	},
	"metrics": {
		"greedy": greedy_scores,
		"vibe": vibe_scores,
	},
	"samples": results,
}

with open("2.yaml", "w") as file_out:
	yaml.dump(output_data, file_out, sort_keys=False)

print("\n" + "=" * 40)
print(" FINAL RESULTS COMPARISON ")
print("=" * 40)
df_metrics = pd.DataFrame([greedy_scores, vibe_scores], index=["Greedy", "Vibe"])
print(df_metrics.T)

print("\nExperiment complete. Results saved.")
