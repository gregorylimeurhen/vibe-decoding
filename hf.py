# hf.py
import json, random, sys, torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_vibe_matrix(embedding_weight):
	embedding_norm = embedding_weight.norm(dim=1, keepdim=True).clamp(min=1e-12)
	return embedding_weight / embedding_norm

def prepare_input_text(tokenizer, text):
	if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
		messages = [{"role": "user", "content": text}]
		return tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)
	return text

def vibe_decode(next_token_logits, vibe_matrix):
	probabilities = next_token_logits.softmax(dim=-1)
	return (
		next_token_logits
		+ probabilities @ vibe_matrix @ vibe_matrix.T
	).argmax(dim=-1)

def generate_once(model, tokenizer, vibe_matrix, text, max_new_tokens, use_vibe, device):
	input_text = prepare_input_text(tokenizer, text)
	batch = tokenizer(input_text, return_tensors="pt").to(device)
	input_ids = batch["input_ids"]
	with torch.no_grad():
		for _ in range(max_new_tokens):
			outputs = model(input_ids=input_ids)
			step_logits = outputs.logits[:, -1, :]
			if use_vibe:
				next_ids = vibe_decode(step_logits, vibe_matrix)
			else:
				next_ids = step_logits.argmax(dim=-1)
			input_ids = torch.cat([input_ids, next_ids.unsqueeze(-1)], dim=-1)
	return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
	model_name = input("Model ID: ").strip()
	if not model_name:
		return
	dataset_name = input("Dataset ID: ").strip()
	if not dataset_name:
		return
	config_name = input("Dataset configuration (leave empty if none): ").strip()
	split_name = input("Dataset split: ").strip()
	column_name = input("Dataset input column: ").strip()
	target_column = input("Dataset target column (leave empty if none): ").strip()
	mode = input("Decoding mode (enter 'vibe' or 'vanilla'): ").strip().lower()
	max_new_tokens = int(input("Maximum number of output tokens: ").strip())
	seed = int(input("Seed: ").strip())

	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	try:
		torch.use_deterministic_algorithms(True)
	except Exception:
		pass

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype=torch.bfloat16,
		device_map="auto",
	)
	model.eval()
	device = next(model.parameters()).device

	use_vibe = mode == "vibe"
	vibe_matrix = None
	if use_vibe:
		embedding_weight = model.get_input_embeddings().weight.detach()
		vibe_matrix = build_vibe_matrix(embedding_weight)

	if config_name:
		dataset = load_dataset(dataset_name, config_name, split=split_name)
	else:
		dataset = load_dataset(dataset_name, split=split_name)

	num_examples = len(dataset)
	log_records = []

	for index in tqdm(range(num_examples), desc="Evaluating"):
		example = dataset[index]
		text = str(example[column_name])
		output = generate_once(
			model,
			tokenizer,
			vibe_matrix,
			text,
			max_new_tokens,
			use_vibe,
			device,
		)
		print("INPUT:", text)
		print("OUTPUT:", output)
		print()
		log_entry = {
			"index": index,
			"input": text,
			"output": output,
		}
		if target_column:
			log_entry["target"] = str(example[target_column])
		log_records.append(log_entry)

	with open("hf.json", "w", encoding="utf-8") as log_file:
		json.dump(log_records, log_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
	main()
