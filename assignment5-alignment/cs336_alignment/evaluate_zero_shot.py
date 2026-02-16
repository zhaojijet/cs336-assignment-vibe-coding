import json
import os
import typer
from typing import Callable, List
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

app = typer.Typer()


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    # Statistics for analysis
    correct_and_format = 0
    format_but_wrong = 0
    wrong_format = 0
    format_but_wrong_answer0 = 0  # Format 1, Answer 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        scores = reward_fn(generated_text, ground_truth)

        # Log counts
        r = scores["reward"]
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)

        if fr == 1.0 and ar == 1.0:
            correct_and_format += 1
        elif fr == 1.0 and ar == 0.0:
            format_but_wrong_answer0 += 1
        elif fr == 0.0:
            wrong_format += 1

        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "ground_truth": ground_truth,
                "scores": scores,
            }
        )

    accuracy = total_reward / len(prompts)  # Assuming reward is essentially accuracy
    format_accuracy = total_format_reward / len(prompts)
    answer_accuracy = total_answer_reward / len(prompts)

    print(f"Evaluation Complete.")
    print(f"Average Reward (Accuracy): {accuracy:.4f}")
    print(f"Format Reward: {format_accuracy:.4f}")
    print(f"Answer Reward: {answer_accuracy:.4f}")
    print(f"Detailed Counts:")
    print(f"  Format=1, Answer=1: {correct_and_format}")
    print(f"  Format=1, Answer=0: {format_but_wrong_answer0}")
    print(f"  Format=0: {wrong_format}")

    # Serialize
    with open(output_path, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    # Also save summary metrics
    summary_path = output_path.replace(".jsonl", "_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "format_accuracy": format_accuracy,
                "answer_accuracy": answer_accuracy,
                "total_examples": len(prompts),
            },
            f,
            indent=2,
        )

    print(f"Results saved to {output_path}")


@app.command()
def main(
    model_path: str = "data/models/Qwen2.5-Math-1.5B",
    data_path: str = "data/MATH/validation.jsonl",
    output_path: str = "zero_shot_results.jsonl",
    seed: int = 42,
    limit: int = -1,
):
    print(f"Loading model from {model_path}...")
    llm = LLM(model=model_path, seed=seed)

    # Load data
    print(f"Loading data from {data_path}...")
    prompts = []
    ground_truths = []

    # Load prompt template
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    with open(data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            question = example.get("problem") or example.get("question")
            answer = example.get("solution") or example.get("answer")

            formatted_prompt = prompt_template.format(question=question)
            prompts.append(formatted_prompt)
            ground_truths.append(answer)

            if limit > 0 and len(prompts) >= limit:
                break

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,  # As per PDF
    )

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
