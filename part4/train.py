"""
Part 4, Step 2 — SageMaker training script using HuggingFace PEFT/LoRA.
Fine-tunes distilgpt2 on financial Q&A data; generalizes to any causal LM.

Run locally:   python train.py --model-dir ./output --training-dir ./data
Run on SM:     Launched by launch_training_job() at bottom of this file.

Requirements:
    pip install transformers peft datasets torch accelerate sagemaker

COMPLIANCE NOTICE — Operator Responsibility:
  This script processes financial domain training data. Before using with real
  customer data, operators MUST:
  1. Scrub all PII/PHI from training examples (names, SSNs, account numbers, DOBs).
  2. Verify a lawful basis exists under GDPR Article 5 (purpose limitation) and
     GLBA if US financial data is involved, before using customer data for training.
  3. Ensure SageMaker training jobs run in a VPC with no internet access if data
     is customer-sensitive (SageMaker network isolation mode).
  The synthetic Q&A pairs in prepare_dataset.py contain no real customer data and
  are safe to use as-is. See prepare_dataset.py for the full compliance notice.
  Operators are solely responsible for production training data compliance.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

BASE_MODEL = "distilgpt2"

EVAL_PROMPTS = [
    "Human: What does KYC stand for and what information does it require?\n\nAssistant:",
    "Human: What is a Suspicious Activity Report and when must it be filed?\n\nAssistant:",
    "Human: If I invest $5,000 at 6% APY compounded monthly for 3 years, approximate the final amount.\n\nAssistant:",
    "Human: What is the difference between APR and APY?\n\nAssistant:",
    "Human: What does FDIC insurance cover and what is the limit?\n\nAssistant:",
    "Human: What is the 2024 401(k) contribution limit for employees under 50?\n\nAssistant:",
    "Human: Name three common red flags for money laundering.\n\nAssistant:",
    "Human: What is the key difference between a Roth IRA and a Traditional IRA?\n\nAssistant:",
    "Human: What is mortgage LTV ratio and why does lender care about it?\n\nAssistant:",
    "Human: How does credit utilization ratio affect a FICO credit score?\n\nAssistant:",
]


class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_metrics: list[dict] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.epoch_metrics.append(
                {
                    "epoch": state.epoch,
                    "step": state.global_step,
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Financial Q&A LoRA fine-tuning")
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./output"))
    p.add_argument("--training-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./data"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    return p.parse_args()


def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                records.append({"text": obj["prompt"] + obj["completion"]})
    return Dataset.from_list(records)


def tokenize_batch(examples: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def compute_perplexity(loss: float) -> float:
    return round(math.exp(loss), 4)


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return generated.strip()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    training_dir = Path(args.training_dir)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_file = training_dir / "train.jsonl"
    val_file = training_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    train_ds = load_jsonl_dataset(str(train_file))
    val_ds = (
        load_jsonl_dataset(str(val_file))
        if val_file.exists()
        else train_ds.select(range(min(5, len(train_ds))))
    )

    def tok_fn(batch):
        return tokenize_batch(batch, tokenizer, args.max_length)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

    # ── Model + LoRA ──────────────────────────────────────────────────────────
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # GPT-2 attention projection; adjust per model family
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    metrics_cb = MetricsCallback()

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[metrics_cb],
    )

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"\nModel saved to {model_dir}")

    # ── Evaluate on held-out prompts ──────────────────────────────────────────
    model.config_eval = model.eval()
    sample_outputs = []
    print("\nSample outputs on financial compliance prompts:")
    for prompt in EVAL_PROMPTS:
        t0 = time.time()
        output = generate_sample(model, tokenizer, prompt)
        latency = round(time.time() - t0, 3)
        sample_outputs.append({"prompt": prompt, "output": output, "latency_s": latency})
        question = prompt.split("Human: ")[1].split("\n")[0][:60]
        print(f"  Q: {question}")
        print(f"  A: {output[:120]}\n")

    # ── Report ────────────────────────────────────────────────────────────────
    final = metrics_cb.epoch_metrics[-1] if metrics_cb.epoch_metrics else {}
    train_loss = final.get("eval_loss", float("nan"))
    perplexity = compute_perplexity(train_loss) if not math.isnan(train_loss) else None

    report = {
        "base_model": BASE_MODEL,
        "lora_config": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["c_attn"]},
        "training_args": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "epoch_metrics": metrics_cb.epoch_metrics,
        "final_eval_loss": train_loss,
        "final_perplexity": perplexity,
        "sample_outputs": sample_outputs,
    }

    report_path = model_dir / "training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Training metrics written to {report_path}")
    print(f"Final eval loss: {train_loss:.4f} | Perplexity: {perplexity}")


if __name__ == "__main__":
    main()


# ── SageMaker launcher — call this from a separate script or notebook ─────────


def launch_training_job(
    role_arn: str,
    bucket: str,
    job_name: str | None = None,
    instance_type: str = "ml.g4dn.xlarge",
    instance_count: int = 1,
) -> str:
    """
    Submit a SageMaker training job for financial Q&A LoRA fine-tuning.

    Args:
        role_arn:       IAM role ARN with SageMaker + S3 + ECR permissions.
        bucket:         S3 bucket where training data lives (financial-qa/ prefix).
        job_name:       Optional name; auto-generated if not provided.
        instance_type:  GPU instance (g4dn.xlarge = T4, cheapest GPU-backed).
        instance_count: Number of training instances.

    Returns:
        The SageMaker training job name.
    """
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    job_name = job_name or f"financial-qa-lora-{int(time.time())}"

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=str(Path(__file__).parent),
        role=role_arn,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        instance_type=instance_type,
        instance_count=instance_count,
        hyperparameters={
            "epochs": 5,
            "learning-rate": 2e-4,
            "batch-size": 8,
            "max-length": 256,
        },
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"'loss': ([0-9\.]+)"},
            {"Name": "eval:loss", "Regex": r"'eval_loss': ([0-9\.]+)"},
        ],
        volume_size=20,
        max_run=3600,
        output_path=f"s3://{bucket}/training-output/",
    )

    estimator.fit(
        inputs={"training": f"s3://{bucket}/financial-qa/"},
        job_name=job_name,
        wait=False,
    )

    print(f"Submitted: {job_name}")
    print(f"Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    return job_name
