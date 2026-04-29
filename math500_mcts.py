"""
StandardMCTS on MATH500 with vLLM (LLM) + Qwen2.5-Math-PRM-7B (PRM scoring).

GPU layout (2-GPU setup):
  - GPU 0: Qwen2.5-7B-Instruct via vLLM  (generation)
  - GPU 1: Qwen2.5-Math-PRM-7B via transformers  (scoring)

NOTE: vLLM's reward-model interface returns one score per sequence and
cannot extract per-step scores at arbitrary token positions.  The PRM
scoring here therefore uses transformers directly on GPU 1 — this is
the same approach used in baseline-tts/src/reason/inference/infer_fns.py.

Usage:
  python math500_mcts.py \
    --llm_path  /path/to/Qwen2.5-7B-Instruct \
    --prm_path  /path/to/Qwen2.5-Math-PRM-7B \
    --dataset   /path/to/test500.jsonl \
    --output    results/math500_mcts.json \
    --mcts_steps 16 \
    --num_problems 500
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── treequest ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from treequest import StandardMCTS
from treequest.ranker import top_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default model paths (update to match your local copies)
# ─────────────────────────────────────────────────────────────────────────────
HF_CACHE = "/prj/corp/airesearch/lasvegas/vol1-scratch/huggingface_hub_cache"
DEFAULT_LLM_PATH = (
    f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct"
    "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_PRM_PATH = "Qwen/Qwen2.5-Math-PRM-7B"   # overrride with local path
DEFAULT_DATASET  = (
    "/prj/corp/airesearch/lasvegas/vol11-scratch/nluu"
    "/baseline-tts/src/envs/MATH/dataset/test500.jsonl"
)

# System prompt used for both LLM generation and PRM conversation formatting
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Qwen2.5-Math-PRM-7B uses <extra_0> as step-boundary marker (token id 151651)
PRM_STEP_TAG     = "<extra_0>"
PRM_STEP_TAG_ID  = 151651   # verified against tokenizer vocab


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper (vLLM)
# ─────────────────────────────────────────────────────────────────────────────

class LLMGenerator:
    """
    Wraps vLLM LLM for Qwen2.5-7B-Instruct generation.

    Two sampling strategies are exposed via generate():
      - temperature=0.7  ("explore")  — used for action "gen_a"
      - temperature=1.0  ("diverse")  — used for action "gen_b"

    Both generate a complete step-by-step solution from the problem.
    parent_solution is accepted but ignored; each generation is
    independent so that the MCTS tree explores a diverse solution space.
    """

    def __init__(self, model_path: str, gpu_id: int = 0,
                 max_tokens: int = 2048, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams  # deferred import

        logger.info(f"Loading LLM from {model_path} onto GPU {gpu_id} …")
        # vLLM picks the first visible GPU; we restrict visibility before import
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            device=f"cuda:{gpu_id}",
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = max_tokens
        logger.info("LLM ready.")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sample(self, prompt: str, temperature: float) -> str:
        from vllm import SamplingParams  # deferred import

        params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    # ── public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        problem: str,
        parent_solution: Optional[str] = None,  # noqa: ARG002  (ignored)
        temperature: float = 0.7,
    ) -> str:
        prompt = self._build_prompt(problem)
        return self._sample(prompt, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# PRM wrapper (transformers, GPU 1)
# ─────────────────────────────────────────────────────────────────────────────

class QwenPRMScorer:
    """
    Wraps Qwen2.5-Math-PRM-7B for step-level reward scoring.

    The model has a 2-class output head (bad=0, good=1).
    Each step is terminated with <extra_0>; the model emits a probability
    P(good | step) at that position.  We return the score of the *last*
    step as the overall solution quality signal.

    Reference implementation:
      baseline-tts/src/reason/inference/infer_fns.py :: _qwen_infer_fn
    """

    def __init__(self, model_path: str, gpu_id: int = 1):
        logger.info(f"Loading PRM from {model_path} onto GPU {gpu_id} …")
        self.device = torch.device(f"cuda:{gpu_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        logger.info("PRM ready.")

    # ── step parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_steps(solution: str) -> list[str]:
        """
        Split a solution text into a list of step strings.

        Handles several common formats:
          • "Step 1: …\nStep 2: …"
          • "**Step 1** …\n\n**Step 2** …"
          • Paragraph-separated (blank line between steps)
          • Plain sentences ending with ".\n"
        Falls back to treating the entire solution as one step.
        """
        # 1. Named steps: "Step N:" or "**Step N"
        named = re.split(r"(?:^|\n)(?:\*{0,2}Step\s+\d+[:\.]?\*{0,2})", solution)
        named = [s.strip() for s in named if s.strip()]
        if len(named) >= 2:
            return named

        # 2. Double-newline paragraphs
        paras = [p.strip() for p in re.split(r"\n{2,}", solution) if p.strip()]
        if len(paras) >= 2:
            return paras

        # 3. Single-newline sentences
        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        if len(lines) >= 2:
            return lines

        return [solution.strip()]

    # ── conversation builder ───────────────────────────────────────────────────

    def _build_conversation(self, problem: str, solution: str) -> list[dict]:
        """
        Format input for the PRM following the Qwen2.5-Math-PRM convention:
          system + user(problem) + assistant(step1<extra_0>step2<extra_0>…)
        """
        steps = self._parse_steps(solution)
        assistant_content = "".join(f"{step}{PRM_STEP_TAG}" for step in steps)
        return [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": assistant_content},
        ]

    # ── scoring ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def score(self, problem: str, solution: str) -> float:
        """
        Return the PRM score of the *last* step ∈ [0, 1].
        Returns 0.0 if the model cannot find any step markers.
        """
        conversation = self._build_conversation(problem, solution)
        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt"
        ).to(self.device)

        # Mark positions of <extra_0> tokens
        step_mask = input_ids == PRM_STEP_TAG_ID   # shape: [1, seq_len]

        if not step_mask.any():
            logger.debug("PRM: no step-boundary tokens found; returning 0.0")
            return 0.0

        # Forward pass — model outputs shape [1, seq_len, 2] (binary head)
        logits = self.model(input_ids)[0]           # [1, seq_len, 2]
        probs  = logits.softmax(dim=-1)[0]          # [seq_len,    2]

        # Extract scores at every <extra_0> position, take the "good" channel (dim 1)
        step_probs  = probs[step_mask[0]]           # [n_steps, 2]
        step_scores = step_probs[:, 1]              # [n_steps]  P(good)

        last_score = float(step_scores[-1].item())
        return max(0.0, min(1.0, last_score))       # clamp to [0, 1]

    @torch.inference_mode()
    def score_batch(
        self, problem: str, solutions: list[str]
    ) -> list[float]:
        """Score a list of solutions for the same problem (sequential)."""
        return [self.score(problem, sol) for sol in solutions]


# ─────────────────────────────────────────────────────────────────────────────
# generate_fn factory
# ─────────────────────────────────────────────────────────────────────────────

def make_generate_fn(
    generator: LLMGenerator,
    prm: QwenPRMScorer,
    problem: str,
    temperature: float,
):
    """
    Returns a generate_fn compatible with treequest's StandardMCTS.

    Signature: (parent_state: str | None) -> (solution: str, score: float)
    """
    def generate_fn(parent_state: Optional[str]) -> tuple[str, float]:
        solution = generator.generate(
            problem,
            parent_solution=parent_state,
            temperature=temperature,
        )
        score = prm.score(problem, solution)
        return solution, score

    return generate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction & correctness
# ─────────────────────────────────────────────────────────────────────────────

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{…} in text."""
    if "boxed" not in text:
        return None
    # Walk the last \boxed occurrence; handle nested braces
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    rest = text[idx + len("\\boxed"):]
    if not rest or rest[0] != "{":
        return None
    depth, buf = 1, []
    for ch in rest[1:]:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(ch)
        else:
            buf.append(ch)
    return "".join(buf).strip() if depth == 0 else None


def _numeric_equal(a: str, b: str) -> bool:
    """Try to compare a and b as floats."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def is_correct(prediction: Optional[str], ground_truth: str) -> bool:
    """
    Lightweight correctness check (no external grader dependency).
    Tries: exact string match → numeric match → sympy symbolic match.
    For heavier evaluation, plug in grader.math_equal from baseline-tts.
    """
    if prediction is None:
        return False
    pred  = prediction.strip()
    truth = ground_truth.strip()
    if pred == truth:
        return True
    if _numeric_equal(pred, truth):
        return True
    # Sympy symbolic equality (best-effort)
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        return simplify(parse_latex(pred) - parse_latex(truth)) == 0
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Single-problem solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_one(
    problem: str,
    ground_truth: str,
    generator: LLMGenerator,
    prm: QwenPRMScorer,
    mcts_steps: int,
    samples_per_action: int,
    exploration_weight: float,
) -> dict:
    """
    Run StandardMCTS on a single problem; return a result dict.
    """
    algo  = StandardMCTS(
        samples_per_action=samples_per_action,
        exploration_weight=exploration_weight,
    )
    state = algo.init_tree()

    generate_fns = {
        # Two actions with different temperatures → diversity in the MCTS tree
        "gen_a": make_generate_fn(generator, prm, problem, temperature=0.7),
        "gen_b": make_generate_fn(generator, prm, problem, temperature=1.0),
    }

    for _ in range(mcts_steps):
        state = algo.step(state, generate_fns)

    # Best solution = highest-scoring leaf
    pairs = algo.get_state_score_pairs(state)
    if not pairs:
        return {
            "problem":      problem,
            "ground_truth": ground_truth,
            "best_solution": None,
            "best_score":    0.0,
            "predicted_answer": None,
            "is_correct":    False,
            "all_solutions": [],
        }

    best_solution, best_score = max(pairs, key=lambda x: x[1])
    predicted_answer = extract_boxed_answer(best_solution) if best_solution else None
    correct = is_correct(predicted_answer, ground_truth)

    # Keep top-3 for inspection
    top3 = top_k(state, algo, k=3)

    return {
        "problem":          problem,
        "ground_truth":     ground_truth,
        "best_solution":    best_solution,
        "best_score":       best_score,
        "predicted_answer": predicted_answer,
        "is_correct":       correct,
        "top3": [
            {"solution": sol, "score": sc}
            for sol, sc in top3
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="StandardMCTS on MATH500")
    p.add_argument("--llm_path",  default=DEFAULT_LLM_PATH,
                   help="Path to Qwen2.5-7B-Instruct")
    p.add_argument("--prm_path",  default=DEFAULT_PRM_PATH,
                   help="Path to Qwen2.5-Math-PRM-7B")
    p.add_argument("--dataset",   default=DEFAULT_DATASET,
                   help="Path to test500.jsonl")
    p.add_argument("--output",    default="results/math500_mcts.json",
                   help="Output JSON file")
    p.add_argument("--mcts_steps",       type=int,   default=16)
    p.add_argument("--samples_per_action", type=int, default=2)
    p.add_argument("--exploration_weight", type=float, default=1.414)
    p.add_argument("--num_problems", type=int, default=500,
                   help="Number of problems to evaluate (0 = all)")
    p.add_argument("--start_idx",  type=int, default=0)
    p.add_argument("--llm_gpu",   type=int, default=0)
    p.add_argument("--prm_gpu",   type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=2048)
    return p.parse_args()


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    args = parse_args()

    # ── output dir ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    generator = LLMGenerator(
        model_path=args.llm_path,
        gpu_id=args.llm_gpu,
        max_tokens=args.max_tokens,
    )
    prm = QwenPRMScorer(
        model_path=args.prm_path,
        gpu_id=args.prm_gpu,
    )

    # ── load dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset(args.dataset)
    end_idx = (args.start_idx + args.num_problems
               if args.num_problems > 0 else len(dataset))
    subset  = dataset[args.start_idx:end_idx]
    logger.info(f"Evaluating {len(subset)} problems (idx {args.start_idx}–{end_idx-1})")

    # ── run MCTS per problem ──────────────────────────────────────────────────
    results    = []
    n_correct  = 0

    for i, sample in enumerate(tqdm(subset, desc="MCTS", unit="problem")):
        problem      = sample["problem"]
        ground_truth = sample["answer"]
        unique_id    = sample.get("unique_id", f"{args.start_idx + i}")

        logger.info(f"[{i+1}/{len(subset)}] {unique_id}")

        try:
            result = solve_one(
                problem=problem,
                ground_truth=ground_truth,
                generator=generator,
                prm=prm,
                mcts_steps=args.mcts_steps,
                samples_per_action=args.samples_per_action,
                exploration_weight=args.exploration_weight,
            )
        except Exception as e:
            logger.error(f"  Error on {unique_id}: {e}", exc_info=True)
            result = {
                "problem": problem, "ground_truth": ground_truth,
                "best_solution": None, "best_score": 0.0,
                "predicted_answer": None, "is_correct": False,
                "top3": [], "error": str(e),
            }

        result["unique_id"] = unique_id
        results.append(result)
        n_correct += int(result["is_correct"])

        accuracy = n_correct / len(results) * 100
        logger.info(
            f"  predicted={result['predicted_answer']}  "
            f"gt={ground_truth}  "
            f"correct={result['is_correct']}  "
            f"running_acc={accuracy:.1f}%"
        )

    # ── summary ───────────────────────────────────────────────────────────────
    accuracy = n_correct / len(results) * 100 if results else 0.0
    summary = {
        "timestamp":     datetime.now().isoformat(),
        "num_problems":  len(results),
        "num_correct":   n_correct,
        "accuracy":      f"{accuracy:.2f}%",
        "config": {
            "llm_path":           args.llm_path,
            "prm_path":           args.prm_path,
            "mcts_steps":         args.mcts_steps,
            "samples_per_action": args.samples_per_action,
            "exploration_weight": args.exploration_weight,
            "max_tokens":         args.max_tokens,
        },
    }

    output_data = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Accuracy : {accuracy:.2f}%  ({n_correct}/{len(results)})")
    logger.info(f"Saved to : {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
