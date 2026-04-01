"""
Benchmark runner — evaluates all 4 routing configurations against the question set.
Usage: python -m benchmark.run_benchmark [--config CONFIG] [--model MODEL] [--output results.json] [--no-judge]

Runs each question through all 4 configs (or a single config) and records:
  - The answer returned
  - Tools used
  - Execution time
  - Soft match (substring/numeric)
  - LLM-as-judge score (0-5) for answer quality
"""
import json, os, sys, time, re
from pathlib import Path

# Add parent dir to path so we can import agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import run_question, get_llm, CONFIGS


def load_questions():
    qpath = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(qpath, encoding="utf-8") as f:
        return json.load(f)


def soft_match(expected: str, actual: str) -> bool:
    """Check if the expected answer appears somewhere in the actual response."""
    if not expected or not actual:
        return False
    exp_lower = expected.lower().strip()
    act_lower = actual.lower().strip()

    # Direct substring match
    if exp_lower in act_lower:
        return True

    # Try matching key terms (split by / for alternatives)
    alternatives = [a.strip() for a in exp_lower.split("/")]
    for alt in alternatives:
        if alt in act_lower:
            return True

    # Try numeric match (within 5% tolerance)
    try:
        exp_num = float(re.sub(r"[^0-9.\-]", "", exp_lower))
        # Find numbers in actual
        nums = re.findall(r"[\d,]+\.?\d*", act_lower)
        for n in nums:
            act_num = float(n.replace(",", ""))
            if exp_num != 0 and abs(act_num - exp_num) / abs(exp_num) < 0.05:
                return True
            if exp_num == 0 and act_num == 0:
                return True
    except (ValueError, ZeroDivisionError):
        pass

    return False


def check_tool_match(expected_tool: str, tools_used: list) -> bool:
    """Check if the expected tool was used."""
    if expected_tool == "none":
        return len(tools_used) == 0
    if expected_tool == "rag_retrieval":
        return "rag_retrieval" in tools_used
    return expected_tool in tools_used


# ── LLM-as-Judge ──

JUDGE_SYSTEM = """You are an expert evaluator of election data question-answering systems.
Score the answer on a scale of 0-5 based on correctness and completeness.

Scoring rubric:
5 = Perfect: answer is factually correct, precise, and complete
4 = Good: answer is mostly correct with minor imprecisions
3 = Acceptable: answer captures the right direction but has notable gaps or errors
2 = Partial: answer has some correct elements but significant errors
1 = Poor: answer is mostly wrong or misleading
0 = Wrong: answer is completely incorrect, irrelevant, or the system failed to answer

Return ONLY a JSON object with two keys:
- "score": integer 0-5
- "reason": one sentence explaining the score

Example: {"score": 4, "reason": "Correctly identified Biden as the winner but rounded the vote count."}"""


def llm_judge(question: str, expected: str, actual: str, judge_llm) -> dict:
    """Use an LLM to score the answer quality."""
    if not actual:
        return {"score": 0, "reason": "No answer produced"}

    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual[:800]}\n\n"
        f"Score the actual answer (0-5):"
    )

    try:
        resp = judge_llm.invoke([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        # Parse JSON from response
        text = resp.content.strip()
        # Handle markdown code fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        result = json.loads(text)
        return {
            "score": int(result.get("score", 0)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": -1, "reason": f"Judge error: {e}"}


def run_benchmark(configs=None, model="gpt-4o-mini", output_path=None, use_judge=True):
    questions = load_questions()
    configs = configs or list(CONFIGS.keys())

    # Initialize judge LLM (use same model as the agent)
    judge_llm = get_llm(model=model) if use_judge else None

    results = []
    summary = {config: {
        "total": 0, "soft_match": 0, "tool_match": 0, "errors": 0,
        "avg_time": 0, "avg_judge_score": 0, "judge_scores": [],
    } for config in configs}

    judge_label = " + LLM-as-judge" if use_judge else ""
    print(f"Running {len(questions)} questions x {len(configs)} configs = {len(questions) * len(configs)} evaluations{judge_label}")
    print(f"Model: {model}\n")

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        expected_tool = q["expected_tool"]
        category = q["category"]
        dataset = q["dataset"]

        print(f"Q{qid}: {question[:80]}...")

        for config in configs:
            try:
                start = time.time()
                result = run_question(question, config=config, model=model)
                elapsed = time.time() - start

                answer = result.get("answer", "")
                tools_used = result.get("tools_used", [])

                matched = soft_match(expected, answer)
                tool_matched = check_tool_match(expected_tool, tools_used)

                # LLM-as-judge scoring
                judge_result = {"score": -1, "reason": "judge disabled"}
                if use_judge and judge_llm:
                    judge_result = llm_judge(question, expected, answer, judge_llm)

                entry = {
                    "id": qid,
                    "question": question,
                    "category": category,
                    "dataset": dataset,
                    "config": config,
                    "expected_answer": expected,
                    "actual_answer": answer[:500],
                    "expected_tool": expected_tool,
                    "tools_used": tools_used,
                    "soft_match": matched,
                    "tool_match": tool_matched,
                    "judge_score": judge_result["score"],
                    "judge_reason": judge_result["reason"],
                    "time_seconds": round(elapsed, 2),
                    "error": None,
                }
                results.append(entry)

                summary[config]["total"] += 1
                summary[config]["soft_match"] += int(matched)
                summary[config]["tool_match"] += int(tool_matched)
                summary[config]["avg_time"] += elapsed
                if judge_result["score"] >= 0:
                    summary[config]["judge_scores"].append(judge_result["score"])

                judge_str = f" judge={judge_result['score']}/5" if judge_result["score"] >= 0 else ""
                status = "MATCH" if matched else "MISS"
                print(f"  [{config}] {status}{judge_str} ({elapsed:.1f}s) tools={tools_used}")

            except Exception as e:
                results.append({
                    "id": qid,
                    "question": question,
                    "category": category,
                    "dataset": dataset,
                    "config": config,
                    "expected_answer": expected,
                    "actual_answer": None,
                    "expected_tool": expected_tool,
                    "tools_used": [],
                    "soft_match": False,
                    "tool_match": False,
                    "judge_score": 0,
                    "judge_reason": f"Error: {e}",
                    "time_seconds": 0,
                    "error": str(e),
                })
                summary[config]["total"] += 1
                summary[config]["errors"] += 1
                print(f"  [{config}] ERROR: {e}")

    # Compute average judge scores
    for config in configs:
        scores = summary[config]["judge_scores"]
        summary[config]["avg_judge_score"] = round(sum(scores) / len(scores), 2) if scores else 0

    # Print summary
    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY")
    print("=" * 85)
    header = f"{'Config':<20} {'Total':>6} {'Match':>6} {'Match%':>7} {'ToolOK':>7} {'Judge':>7} {'Errors':>7} {'AvgTime':>8}"
    print(header)
    print("-" * 85)

    for config in configs:
        s = summary[config]
        total = s["total"]
        if total > 0:
            match_pct = s["soft_match"] / total * 100
            avg_time = s["avg_time"] / total
        else:
            match_pct = 0
            avg_time = 0
        judge_avg = s["avg_judge_score"]
        print(f"{config:<20} {total:>6} {s['soft_match']:>6} {match_pct:>6.1f}% {s['tool_match']:>7} {judge_avg:>6.1f}/5 {s['errors']:>7} {avg_time:>7.1f}s")

    # Category breakdown
    print("\n" + "=" * 85)
    print("BY CATEGORY (soft_match% | avg judge score)")
    print("=" * 85)
    for cat in ["factual", "numerical", "multi_step", "coalition"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        print(f"\n{cat.upper()} ({len(cat_results)} evaluations):")
        for config in configs:
            config_results = [r for r in cat_results if r["config"] == config]
            if not config_results:
                continue
            matches = sum(1 for r in config_results if r["soft_match"])
            total = len(config_results)
            scores = [r["judge_score"] for r in config_results if r["judge_score"] >= 0]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"  {config:<20} {matches}/{total} ({matches/total*100:>3.0f}%) | judge: {avg_score:.1f}/5")

    # Dataset breakdown
    print("\n" + "=" * 85)
    print("BY DATASET (soft_match% | avg judge score)")
    print("=" * 85)
    for ds in ["us", "israel", "both"]:
        ds_results = [r for r in results if r["dataset"] == ds]
        if not ds_results:
            continue
        print(f"\n{ds.upper()} ({len(ds_results)} evaluations):")
        for config in configs:
            config_results = [r for r in ds_results if r["config"] == config]
            if not config_results:
                continue
            matches = sum(1 for r in config_results if r["soft_match"])
            total = len(config_results)
            scores = [r["judge_score"] for r in config_results if r["judge_score"] >= 0]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"  {config:<20} {matches}/{total} ({matches/total*100:>3.0f}%) | judge: {avg_score:.1f}/5")

    # Clean up summary for JSON serialization
    for config in configs:
        del summary[config]["judge_scores"]

    # Save results
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": results,
            "model": model,
            "configs": configs,
            "judge_enabled": use_judge,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run election agent benchmark")
    parser.add_argument("--config", type=str, default=None,
                        help="Single config to test (default: all)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge scoring (faster, cheaper)")
    args = parser.parse_args()

    configs = [args.config] if args.config else None
    run_benchmark(configs=configs, model=args.model, output_path=args.output, use_judge=not args.no_judge)
