import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# # ==============================================================================
# BENCHMARK CONFIGURATION (Local RTX 2050)
# ==============================================================================
DRY_RUN = True  

MODEL_NAME = "google/flan-t5-base"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pointing to the 100-sample CSV for speed testing
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_100_sample.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "flan_t5_lora_results")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_lora_model")

def train():
    # Detect GPU hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Starting Training Script | Mode: {'DRY RUN (Local)' if DRY_RUN else 'FULL TRAIN (Colab)'}")
    print(f"Compute Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Data File Path: {DATA_PATH}")
    print("=" * 60)

    # 1. Verify Dataset existence
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Please run 'python src/preprocess.py' first!"
        )

    # 2. Load Dataset
    dataset = load_dataset("csv", data_files={"train": DATA_PATH})
    print(f"Loaded {len(dataset['train'])} records for training.")

    # 3. Load Tokenizer
    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization Preprocessing Function
    def preprocess_function(examples):
        inputs = ["summarize: " + str(doc) for doc in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)
        
        labels = tokenizer(text_target=[str(t) for t in examples["target"]], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset["train"].map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # 4. Load Pretrained Base Model & Apply PEFT/LoRA
    print(f"Loading Base Model '{MODEL_NAME}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q", "v"]
    )
    model = get_peft_model(model, peft_config)
    
    print("\n--- Trainable Parameters Summary ---")
    model.print_trainable_parameters()
    print("-" * 36 + "\n")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,        # Optimal batch size for 4GB VRAM
    gradient_accumulation_steps=2,        # Simulates batch size of 8
    num_train_epochs=1,                   # 1 epoch for pure benchmarking
    learning_rate=5e-4,
    fp16=True,                            # Uses RTX 2050 Tensor Cores
    logging_steps=5,
    save_strategy="no",                   # Skip saving disk space during benchmark
    report_to="none"
)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. Execute Training
    print("Execution initialized. Training model...")
    trainer.train()

    # 7. Save LoRA Model Weights and Tokenizer
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Best LoRA Adapter saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train()