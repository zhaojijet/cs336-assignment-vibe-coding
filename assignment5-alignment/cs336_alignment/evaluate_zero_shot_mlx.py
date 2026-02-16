import json
import logging
import argparse
from pathlib import Path

from mlx_lm import load, generate
from tqdm import tqdm

from cs336_alignment.drgrpo_grader import grade
from cs336_alignment.metrics import parse_gsm8k_response, parse_mmlu_response

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/MATH/validation.jsonl"
    )
    parser.add_argument(
        "--output-file", type=str, default="zero_shot_mlx_results.jsonl"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load(args.model_path)

    # R1 Zero Prompt Template
    PROMPT_TEMPLATE = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        "User: {instruction}\n"
        "Assistant: <think>"
    )

    data = []
    with open(args.dataset_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    if args.limit > 0:
        data = data[: args.limit]

    logger.info(f"Evaluating on {len(data)} examples")

    results = []
    correct_count = 0

    for item in tqdm(data):
        question = item.get("problem") or item.get("question")
        gold_answer = item.get("solution") or item.get("answer")

        prompt = PROMPT_TEMPLATE.format(instruction=question)

        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False
        )

        # Parse and Grade
        # The prompt ends with <think>, so response should start with reasoning
        # We need to reconstruct full response for parser or just pass response
        # Our grader expects full response usually or just answer part.
        # But let's follow the standard pattern.

        # Extract <answer> part
        if "<answer>" in response and "</answer>" in response:
            answer_content = response.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            answer_content = response

        is_correct = grade(answer_content, gold_answer)
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": question,
                "gold_answer": gold_answer,
                "response": response,
                "is_correct": is_correct,
            }
        )

    accuracy = correct_count / len(data)
    logger.info(f"Accuracy: {accuracy:.4f}")

    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
