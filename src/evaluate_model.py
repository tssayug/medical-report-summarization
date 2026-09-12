import os
import torch
import pandas as pd
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = os.path.join(BASE_DIR, "outputs", "flan_t5_lora_results", "best_lora_model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_500.csv")

def run_evaluation(num_samples=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device.upper()}")

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Could not find adapter folder at '{ADAPTER_PATH}'!")

    # 1. Load subset of test data for fast local scoring
    df = pd.read_csv(TEST_DATA_PATH).head(num_samples)

    # 2. Load Base Model + Trained LoRA Adapter
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    # Hugging Face evaluate library call
    rouge = evaluate.load("rouge")
    predictions, references = [], []

    print(f"Generating summaries for {num_samples} test cases...")
    for idx, row in df.iterrows():
        prompt = "summarize: " + str(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, num_beams=2)
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(str(row["target"]))

    # 3. Calculate ROUGE Scores
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\n" + "=" * 50)
    print("EVALUATION METRICS (ROUGE SCORES)")
    print("=" * 50)
    for metric, score in results.items():
        print(f" -> {metric.upper()}: {score * 100:.2f}%")

    # 4. Print Sample Output for Verification
    print("\n" + "=" * 50)
    print("SAMPLE CLINICAL OUTPUT COMPARISON")
    print("=" * 50)
    print(f"REPORT INPUT:\n{df.iloc[0]['input'][:300]}...\n")
    print(f"GROUND TRUTH SUMMARY:\n{references[0]}\n")
    print(f"MODEL GENERATED SUMMARY:\n{predictions[0]}")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation(num_samples=50)