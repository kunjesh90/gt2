import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--merge_and_unload", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.merge_and_unload:
        model = model.merge_and_unload()

    model.eval()

    full_prompt = f"{args.prompt}\n\nAnswer:"
    inputs = tok(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tok.eos_token_id,
        )

    output = tok.decode(gen[0], skip_special_tokens=True)
    # Return only the completion after 'Answer:' if present
    if "Answer:" in output:
        output = output.split("Answer:", 1)[1].strip()
    print(output)


if __name__ == "__main__":
    main()
