import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

random.seed(42)


@dataclass
class FDScenario:
    scenario_id: str
    principal: float  # currency units
    annual_rate_percent: float  # e.g., 7.5 means 7.5%
    tenure_years: float  # years
    product_type: str  # 'quarterly_payout' | 'yearly_payout' | 'cumulative_quarterly'


def round_money(x: float) -> float:
    return float(f"{x:.2f}")


def quarterly_payout(principal: float, rate_pct: float, tenure_years: float) -> Dict[str, Any]:
    m = 4
    interest_per_period = principal * (rate_pct / 100.0) / m
    periods = int(round(tenure_years * m))
    total_interest = interest_per_period * periods
    maturity_amount = principal  # principal returned; interest paid periodically
    return {
        "payout_per_period": round_money(interest_per_period),
        "periods": periods,
        "total_interest": round_money(total_interest),
        "maturity_amount": round_money(maturity_amount),
    }


def yearly_payout(principal: float, rate_pct: float, tenure_years: float) -> Dict[str, Any]:
    m = 1
    interest_per_period = principal * (rate_pct / 100.0) / m
    periods = int(round(tenure_years * m))
    total_interest = interest_per_period * periods
    maturity_amount = principal
    return {
        "payout_per_period": round_money(interest_per_period),
        "periods": periods,
        "total_interest": round_money(total_interest),
        "maturity_amount": round_money(maturity_amount),
    }


def cumulative_quarterly(principal: float, rate_pct: float, tenure_years: float) -> Dict[str, Any]:
    m = 4
    periods = int(round(tenure_years * m))
    rate_per_period = (rate_pct / 100.0) / m
    maturity_amount = principal * ((1.0 + rate_per_period) ** periods)
    total_interest = maturity_amount - principal
    return {
        "payout_per_period": None,
        "periods": periods,
        "total_interest": round_money(total_interest),
        "maturity_amount": round_money(maturity_amount),
    }


def generate_random_scenarios(n: int) -> List[FDScenario]:
    scenarios: List[FDScenario] = []
    for i in range(n):
        product_type = random.choices(
            ["quarterly_payout", "yearly_payout", "cumulative_quarterly"],
            weights=[0.4, 0.2, 0.4],
            k=1,
        )[0]
        principal = random.choice([
            random.uniform(5_000, 200_000),
            random.uniform(200_000, 1_000_000),
            random.uniform(1_000_000, 5_000_000),
        ])
        principal = round_money(principal)
        annual_rate_percent = round(random.uniform(3.0, 12.0), 2)

        if product_type == "yearly_payout":
            years = random.choice([1, 2, 3, 5, 7, 10])
        else:
            # multiples of quarter for quarterly products
            quarters = random.choice([2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40])
            years = quarters / 4.0

        tenure_years = float(years)
        scenarios.append(
            FDScenario(
                scenario_id=f"ex_{i:06d}",
                principal=principal,
                annual_rate_percent=annual_rate_percent,
                tenure_years=tenure_years,
                product_type=product_type,
            )
        )
    return scenarios


def build_prompt_and_answer(scn: FDScenario) -> Tuple[str, Dict[str, Any]]:
    if scn.product_type == "quarterly_payout":
        result = quarterly_payout(scn.principal, scn.annual_rate_percent, scn.tenure_years)
        instr = (
            "You are a precise finance calculator. "
            "Compute fixed deposit interest for quarterly payout (non-cumulative). "
            "Return ONLY a compact JSON object with keys maturity_amount, total_interest, payout_per_period."
            f" Principal={scn.principal}, AnnualRatePercent={scn.annual_rate_percent}, TenureYears={scn.tenure_years}."
        )
    elif scn.product_type == "yearly_payout":
        result = yearly_payout(scn.principal, scn.annual_rate_percent, scn.tenure_years)
        instr = (
            "You are a precise finance calculator. "
            "Compute fixed deposit interest for yearly payout (non-cumulative). "
            "Return ONLY a compact JSON object with keys maturity_amount, total_interest, payout_per_period."
            f" Principal={scn.principal}, AnnualRatePercent={scn.annual_rate_percent}, TenureYears={scn.tenure_years}."
        )
    else:
        result = cumulative_quarterly(scn.principal, scn.annual_rate_percent, scn.tenure_years)
        instr = (
            "You are a precise finance calculator. "
            "Compute fixed deposit maturity for cumulative quarterly compounding. "
            "Return ONLY a compact JSON object with keys maturity_amount, total_interest, payout_per_period (use null)."
            f" Principal={scn.principal}, AnnualRatePercent={scn.annual_rate_percent}, TenureYears={scn.tenure_years}."
        )

    response_obj = {
        "maturity_amount": result["maturity_amount"],
        "total_interest": result["total_interest"],
        "payout_per_period": result["payout_per_period"],
    }

    # Ensure compact JSON without spaces for easier parsing
    response_json = json.dumps(response_obj, separators=(",", ":"))

    # Supervised fine-tuning expects target text. We'll use a simple instruction-response format.
    prompt = instr.strip()
    target = response_json

    return prompt, response_obj


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(train_size: int = 5000, eval_size: int = 500):
    scenarios = generate_random_scenarios(train_size + eval_size)

    records: List[Dict[str, Any]] = []
    for scn in scenarios:
        prompt, response_obj = build_prompt_and_answer(scn)
        record = {
            "id": scn.scenario_id,
            "product_type": scn.product_type,
            "principal": scn.principal,
            "annual_rate_percent": scn.annual_rate_percent,
            "tenure_years": scn.tenure_years,
            "prompt": prompt,
            "response": json.dumps(response_obj, separators=(",", ":")),
        }
        records.append(record)

    train_records = records[:train_size]
    eval_records = records[train_size: train_size + eval_size]

    write_jsonl("/workspace/data/fd_train.jsonl", train_records)
    write_jsonl("/workspace/data/fd_eval.jsonl", eval_records)
    print(f"Wrote {len(train_records)} train and {len(eval_records)} eval examples.")


if __name__ == "__main__":
    main()
