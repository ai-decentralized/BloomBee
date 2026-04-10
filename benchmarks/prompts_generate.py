import random
import json
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca")["train"]
batch_size = 32
num_groups = 10

random.seed(1)
groups = []
for i in range(num_groups):
    indices = random.sample(range(len(dataset)), batch_size)
    groups.append(indices)

with open("eval_indices.json", "w") as f:
    json.dump(groups, f)

print(f"Generated {num_groups} groups of {batch_size} prompts each")