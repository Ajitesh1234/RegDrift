"""Step 4: LoRA fine-tune a small causal LM as the clause -> obligations extractor (run on a free Colab T4).

    pip install -r requirements-train.txt
    python train_lora.py --epochs 6

Only the completion tokens are trained on (the prompt is masked), so the model learns to emit the JSON.
Outputs ./regdrift-lora/ (adapter + tokenizer + training_meta.json). Use it with LORA_ADAPTER=regdrift-lora.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build_labels(prompt_ids, full_ids):
    """Loss only on the completion: prompt positions get -100 (ignored by the loss)."""
    n = min(len(prompt_ids), len(full_ids))
    return [-100] * n + list(full_ids[n:])


def encode_row(tok, row, max_len=512):
    prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tok(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    full_ids = full_ids[:max_len]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": build_labels(prompt_ids, full_ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct", help="small, ungated, fits a free T4")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "train.jsonl"))
    ap.add_argument("--out", default="regdrift-lora")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments, set_seed)
    if not torch.cuda.is_available():
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU. Continuing on CPU (slow).")
    set_seed(a.seed)
    with open(a.data, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    if len(rows) < 20:
        sys.exit(f"Only {len(rows)} training rows; run build_lora_data.py first.")
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = Dataset.from_list([encode_row(tok, r) for r in rows])
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir="out", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
                             learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=5,
                             save_strategy="no", fp16=torch.cuda.is_available(), report_to="none", seed=a.seed)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "training_meta.json"), "w") as fh:
        json.dump({"base_model": a.base, "train_rows": len(rows), "epochs": a.epochs, "lr": a.lr, "rank": a.rank,
                   "batch": a.batch, "seed": a.seed}, fh, indent=2)
    print(f"Saved adapter to {a.out}/")


if __name__ == "__main__":
    main()
