import argparse
import json
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        # Attempt to extract JSON substring
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return None
        return None


def numbers_close(a: float, b: float, tol: float = 0.02) -> bool:
    # Absolute tolerance of 2 cents by default
    return abs(float(a) - float(b)) <= tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--eval_path", default="/workspace/data/fd_eval.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=torch.cuda.is_available(),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    ) if torch.cuda.is_available() else None

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    # Load eval dataset
    lines = open(args.eval_path, "r", encoding="utf-8").read().splitlines()
    n = len(lines)

    correct = 0
    attempted = 0

    for line in lines:
        row = json.loads(line)
        prompt = row["prompt"]
        expected = json.loads(row["response"])  # string in dataset

        inputs = tok(f"{prompt}\n\nAnswer:", return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        out_text = tok.decode(gen[0], skip_special_tokens=True)
        if "Answer:" in out_text:
            out_text = out_text.split("Answer:", 1)[1].strip()

        pred = try_parse_json(out_text)
        if pred is None:
            continue

        attempted += 1
        keys = ["maturity_amount", "total_interest", "payout_per_period"]
        all_ok = True
        for k in keys:
            exp_v = expected.get(k)
            pred_v = pred.get(k)
            if exp_v is None and pred_v is None:
                continue
            if exp_v is None or pred_v is None:
                all_ok = False
                break
            if pred_v is None:
                pred_v = 0.0
            try:
                if not numbers_close(float(pred_v), float(exp_v)):
                    all_ok = False
                    break
            except Exception:
                all_ok = False
                break
        if all_ok:
            correct += 1

    acc = (correct / attempted) * 100.0 if attempted > 0 else 0.0
    print(json.dumps({"eval_count": n, "attempted": attempted, "correct": correct, "accuracy_percent": round(acc, 2)}, indent=2))


if __name__ == "__main__":
    main()
