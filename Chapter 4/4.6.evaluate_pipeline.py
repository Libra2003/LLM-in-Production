import torch
from transformers import pipeline
from datasets import Dataset, load_dataset
from evaluate import evaluator
import evaluate
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pull model, data, and metrics
pipe = pipeline(
    "text-generation", model="gpt2", pad_token_id=50256, device=device
)
wino_bias = load_dataset("sasha/wino_bias_prompt1", split="test")
poplarity = evaluate.load("regard")
task_evaluator = evaluator("text-generation")

# Prepare dataset
def prepare_dataset(wino_bias, pronoun):
    data = wino_bias.filter(
        lambda example: example["bias_pronoun"] == pronoun
    ).shuffle()
    df = data.to_pandas()
    df["prompts"] = df["prompt_phrase"] + " " + df["bias_Pronoun"]
    return Dataset.from_pandas(df)

female_prompts = prepare_dataset(wino_bias, "she")
male_prompts = prepare_dataset(wino_bias, "he")

# Run through evaluation pipeline
female_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=female_prompts,
    input_column="prompts",
    metric=poplarity,
)
male_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=male_prompts,
    input_column="prompts",
    metric=poplarity,
)

# Analyze results
def flatten_results(results):
    flattened_results=[]
    for result in results["regard"]:
        item_dict = {}
        for item in result:
            item_dict[item["label"]] = item["score"]
        flatten_results.append(item_dict)
    return pd.DataFrame(flatten_results)

print(flatten_results(female_results).mean())
print(flatten_results(male_results).mean())