import os, subprocess, re

gitignore = [line.strip() for line in open(".gitignore") if line.strip() and not line.startswith("#")] if os.path.exists(".gitignore") else []
paths = []

for root, _, files in os.walk("."):
	for name in files:
		if name == "_.py": continue
		if name.endswith(".py") or ".env" in name:
			paths.append(os.path.join(root, name))

ignored = set(subprocess.run(["git", "check-ignore", *paths], capture_output=True, text=True).stdout.splitlines()) if os.path.exists(".git") else set()

with open("_.md", "w") as out:
	for path in sorted(paths):
		if path in ignored and ".env" not in path: continue
		text = open(path).read().rstrip("\n")
		if ".env" in path:
			text = re.sub(r"(?m)(\b\w*API_KEY\s*=\s*)\S+", r"\1foo", text)
		out.write(f"file: `{path}`\n```python\n{text}\n```\n\n")
