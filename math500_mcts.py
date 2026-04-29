"""
StandardMCTS on math reasoning benchmarks.

Supports:
  Datasets : math500 | aime24 | amc23 | minervamath
  LLM      : qwen2.5-7b | qwen2.5-3b | phi-4-mini  (or any HF model ID / local path)
  PRM      : Qwen2.5-Math-PRM-7B  (via transformers on a separate GPU)

GPU layout (2-GPU setup, default):
  GPU 0 — LLM generation  (vLLM)
  GPU 1 — PRM scoring     (transformers)

NOTE on vLLM + PRM:
  vLLM's reward-model interface (task="reward") returns one score per sequence
  and cannot extract per-step scores at arbitrary token positions.  The PRM
  therefore uses transformers directly — the same approach as
  baseline-tts/src/reason/inference/infer_fns.py::_qwen_infer_fn.

Usage examples:
  # MATH500 with Qwen2.5-7B
  python math500_mcts.py --dataset math500 --llm qwen2.5-7b

  # AIME24 with Phi-4-mini, 32 MCTS steps
  python math500_mcts.py --dataset aime24 --llm phi-4-mini --mcts_steps 32

  # Full custom paths
  python math500_mcts.py \\
      --dataset minervamath \\
      --llm_path /path/to/Qwen2.5-3B-Instruct \\
      --prm_path /path/to/Qwen2.5-Math-PRM-7B \\
      --output   results/minerva_qwen3b.json
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── treequest ─────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO / "src"))
from treequest import StandardMCTS
from treequest.ranker import top_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Grader: use baseline-tts grader when available, fall back to lightweight
# ─────────────────────────────────────────────────────────────────────────────
_BASELINE_TTS = (
    "/prj/corp/airesearch/lasvegas/vol11-scratch/nluu/baseline-tts/src"
)
_GRADER_AVAILABLE = False

# Try importing the battle-tested grader from baseline-tts
try:
    sys.path.insert(0, _BASELINE_TTS)
    sys.path.insert(0, os.path.join(_BASELINE_TTS, "envs/MATH"))
    from envs.MATH.grader import math_equal as _math_equal          # type: ignore
    from envs.MATH.parse_utils_qwen import extract_answer as _extract_answer  # type: ignore
    _GRADER_AVAILABLE = True
    logger.info("Using baseline-tts grader (math_equal + extract_answer).")
except Exception as _e:
    logger.warning(f"baseline-tts grader not found ({_e}); using lightweight fallback.")


def _extract_answer_fallback(pred_str: str, data_name: str) -> str:
    """
    Lightweight fallback extractor.
    Priority: \\boxed{} → "The answer is X" → last number in text.
    """
    # 1. \\boxed{...}
    if "boxed" in pred_str:
        idx = pred_str.rfind("\\boxed")
        if idx != -1:
            rest = pred_str[idx + len("\\boxed"):]
            if rest and rest[0] == "{":
                depth, buf = 1, []
                for ch in rest[1:]:
                    if ch == "{":
                        depth += 1; buf.append(ch)
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return "".join(buf).strip()
                        buf.append(ch)
                    else:
                        buf.append(ch)

    # 2. "the answer is X" / "final answer is X"
    for pat in ("he answer is", "final answer is"):
        if pat in pred_str:
            candidate = pred_str.split(pat)[-1].strip().split()[0]
            return candidate.rstrip(".,;")

    # 3. Last number in string
    if data_name != "minerva_math":
        nums = re.findall(r"-?\d*\.?\d+(?:e[+-]?\d+)?", pred_str.replace(",", ""))
        if nums:
            return nums[-1]
    return ""


def _math_equal_fallback(
    prediction: Union[str, float, None],
    reference: Union[str, float, None],
) -> bool:
    """Lightweight fallback: string → numeric → sympy."""
    if prediction is None or reference is None:
        return False
    pred  = str(prediction).strip()
    ref   = str(reference).strip()
    if pred == ref:
        return True
    # Numeric
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except (ValueError, TypeError):
        pass
    # Sympy
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        return simplify(parse_latex(pred) - parse_latex(ref)) == 0
    except Exception:
        pass
    return False


def extract_answer(pred_str: str, data_name: str) -> str:
    if _GRADER_AVAILABLE:
        return _extract_answer(pred_str, data_name)
    return _extract_answer_fallback(pred_str, data_name)


def is_correct(
    prediction: Optional[str],
    ground_truth: Union[str, float],
    data_name: str,
) -> bool:
    if prediction is None or str(prediction).strip() == "":
        return False
    if _GRADER_AVAILABLE:
        return _math_equal(str(prediction).strip(), str(ground_truth).strip())
    return _math_equal_fallback(prediction, ground_truth)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    file:        str   # relative to repo root
    problem_key: str   # field name for the problem/question
    answer_key:  str   # field name for the ground-truth answer
    data_name:   str   # passed to extract_answer / math_equal


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "math500": DatasetConfig(
        file="datasets/test500.jsonl",
        problem_key="problem",
        answer_key="answer",
        data_name="math",
    ),
    "aime24": DatasetConfig(
        file="datasets/test_aime.jsonl",
        problem_key="problem",
        answer_key="extracted_groundtruth",
        data_name="math",          # integer answers, math_equal handles "025" == "25"
    ),
    "amc23": DatasetConfig(
        file="datasets/test_amc.jsonl",
        problem_key="problem",
        answer_key="extracted_groundtruth",
        data_name="math",          # float answers stored as 27.0, 8.0, etc.
    ),
    "minervamath": DatasetConfig(
        file="datasets/test_minerva.jsonl",
        problem_key="question",    # NOTE: "question", not "problem"
        answer_key="answer",
        data_name="minerva_math",  # skip unit stripping in strip_string
    ),
}


def load_dataset(cfg: DatasetConfig, base_dir: Path) -> list[dict]:
    path = base_dir / cfg.file
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
_HF_CACHE = "/prj/corp/airesearch/lasvegas/vol1-scratch/huggingface_hub_cache"

# Known local snapshots (skip HF download when available)
_LOCAL_SNAPSHOTS: dict[str, str] = {
    "Qwen/Qwen2.5-7B-Instruct": (
        f"{_HF_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
}

MODEL_REGISTRY: dict[str, str] = {
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b":  "Qwen/Qwen2.5-3B-Instruct",
    "phi-4-mini":  "microsoft/Phi-4-mini-instruct",
}

DEFAULT_PRM_PATH = "Qwen/Qwen2.5-Math-PRM-7B"

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Qwen2.5-Math-PRM-7B uses <extra_0> as step-boundary token (id=151651)
PRM_STEP_TAG    = "<extra_0>"
PRM_STEP_TAG_ID = 151651


def resolve_model_path(llm_arg: str) -> str:
    """
    Accept either a short registry key ('qwen2.5-7b'), an HF model ID
    ('Qwen/Qwen2.5-7B-Instruct'), or an absolute local path.
    Returns the path/ID to pass to vLLM / transformers.
    """
    # 1. Registry short-name
    if llm_arg in MODEL_REGISTRY:
        hf_id = MODEL_REGISTRY[llm_arg]
        local = _LOCAL_SNAPSHOTS.get(hf_id)
        if local and Path(local).exists():
            logger.info(f"Using local snapshot: {local}")
            return local
        logger.info(f"Using HF ID (will download if needed): {hf_id}")
        return hf_id
    # 2. Known HF ID with local snapshot
    if llm_arg in _LOCAL_SNAPSHOTS:
        local = _LOCAL_SNAPSHOTS[llm_arg]
        if Path(local).exists():
            return local
    # 3. Absolute local path or raw HF ID
    return llm_arg


# ─────────────────────────────────────────────────────────────────────────────
# Token counter
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenStats:
    """Accumulates token usage for one problem across all MCTS steps."""
    llm_prompt_tokens:    int = 0
    llm_generated_tokens: int = 0
    prm_input_tokens:     int = 0
    llm_calls:            int = 0
    prm_calls:            int = 0

    def add_llm(self, prompt_toks: int, gen_toks: int) -> None:
        self.llm_prompt_tokens    += prompt_toks
        self.llm_generated_tokens += gen_toks
        self.llm_calls            += 1

    def add_prm(self, input_toks: int) -> None:
        self.prm_input_tokens += input_toks
        self.prm_calls        += 1

    @property
    def total_llm_tokens(self) -> int:
        return self.llm_prompt_tokens + self.llm_generated_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_llm_tokens + self.prm_input_tokens

    def to_dict(self) -> dict:
        return {
            "llm_prompt_tokens":    self.llm_prompt_tokens,
            "llm_generated_tokens": self.llm_generated_tokens,
            "prm_input_tokens":     self.prm_input_tokens,
            "total_llm_tokens":     self.total_llm_tokens,
            "total_tokens":         self.total_tokens,
            "llm_calls":            self.llm_calls,
            "prm_calls":            self.prm_calls,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper (vLLM)
# ─────────────────────────────────────────────────────────────────────────────
class LLMGenerator:
    """
    vLLM-backed generator for any HuggingFace chat model.

    Two temperature settings are used as separate MCTS actions:
      gen_a — temperature 0.7  (moderate diversity)
      gen_b — temperature 1.0  (high diversity)

    parent_solution is accepted but ignored; each call generates a
    complete, independent solution so the MCTS tree explores diverse
    candidate answers.
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int = 0,
        max_tokens: int = 2048,
        gpu_memory_utilization: float = 0.85,
    ):
        from vllm import LLM  # deferred import

        logger.info(f"Loading LLM: {model_path}  (GPU {gpu_id})")
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

    def _build_prompt(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sample(self, prompt: str, temperature: float) -> tuple[str, int, int]:
        from vllm import SamplingParams  # deferred import

        params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        out         = self.llm.generate([prompt], params)[0]
        text        = out.outputs[0].text.strip()
        prompt_toks = len(out.prompt_token_ids)
        gen_toks    = len(out.outputs[0].token_ids)
        return text, prompt_toks, gen_toks

    def generate(
        self,
        problem: str,
        parent_solution: Optional[str] = None,  # noqa: ARG002 (kept for API compat)
        temperature: float = 0.7,
        token_stats: Optional[TokenStats] = None,
    ) -> str:
        prompt = self._build_prompt(problem)
        text, prompt_toks, gen_toks = self._sample(prompt, temperature)
        if token_stats is not None:
            token_stats.add_llm(prompt_toks, gen_toks)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# PRM wrapper (transformers, separate GPU)
# ─────────────────────────────────────────────────────────────────────────────
class QwenPRMScorer:
    """
    Qwen2.5-Math-PRM-7B step-level reward scorer.

    The model has a binary classification head (bad=0, good=1).
    Each reasoning step is delimited by <extra_0>; P(good | step) is read
    at that token position.  We return the score of the *last* step as the
    overall quality signal for the solution.

    Reference: baseline-tts/.../infer_fns.py :: _qwen_infer_fn
    """

    def __init__(self, model_path: str, gpu_id: int = 1):
        logger.info(f"Loading PRM: {model_path}  (GPU {gpu_id})")
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

    @staticmethod
    def _parse_steps(solution: str) -> list[str]:
        """
        Split solution text into reasoning steps.
        Tries named steps → double-newline paragraphs → single-newline lines.
        """
        named = re.split(r"(?:^|\n)(?:\*{0,2}Step\s+\d+[:.]\*{0,2})", solution)
        named = [s.strip() for s in named if s.strip()]
        if len(named) >= 2:
            return named

        paras = [p.strip() for p in re.split(r"\n{2,}", solution) if p.strip()]
        if len(paras) >= 2:
            return paras

        lines = [l.strip() for l in solution.split("\n") if l.strip()]
        if len(lines) >= 2:
            return lines

        return [solution.strip()] if solution.strip() else [""]

    def _build_conversation(self, problem: str, solution: str) -> list[dict]:
        steps = self._parse_steps(solution)
        assistant_content = "".join(f"{step}{PRM_STEP_TAG}" for step in steps)
        return [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": assistant_content},
        ]

    @torch.inference_mode()
    def score(
        self,
        problem: str,
        solution: str,
        token_stats: Optional[TokenStats] = None,
    ) -> float:
        """
        Return P(last step is correct) ∈ [0, 1].
        Returns 0.0 when no step-boundary tokens are found.
        """
        conversation = self._build_conversation(problem, solution)
        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt"
        ).to(self.device)

        if token_stats is not None:
            token_stats.add_prm(input_ids.shape[1])

        step_mask = (input_ids == PRM_STEP_TAG_ID)   # [1, seq_len]
        if not step_mask.any():
            logger.debug("PRM: no <extra_0> tokens found; returning 0.0")
            return 0.0

        # model → [1, seq_len, 2]  (binary classification head)
        logits      = self.model(input_ids)[0]         # [1, seq_len, 2]
        probs       = logits.softmax(dim=-1)[0]        # [seq_len, 2]
        step_probs  = probs[step_mask[0]]              # [n_steps, 2]
        step_scores = step_probs[:, 1]                 # P(good) per step

        return float(step_scores[-1].clamp(0.0, 1.0).item())


# ─────────────────────────────────────────────────────────────────────────────
# generate_fn factory
# ─────────────────────────────────────────────────────────────────────────────
def make_generate_fn(
    generator: LLMGenerator,
    prm: QwenPRMScorer,
    problem: str,
    temperature: float,
    token_stats: TokenStats,
):
    """
    Returns a treequest-compatible generate_fn:
        (parent_state: str | None) -> (solution: str, score: float)

    token_stats is updated in-place on every call.
    """
    def generate_fn(parent_state: Optional[str]) -> tuple[str, float]:
        solution = generator.generate(
            problem,
            parent_solution=parent_state,
            temperature=temperature,
            token_stats=token_stats,
        )
        score = prm.score(problem, solution, token_stats=token_stats)
        return solution, score

    return generate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one solution
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_answer(
    solution: str,
    ground_truth: Any,
    data_name: str,
) -> tuple[Optional[str], bool]:
    """
    Extract the predicted answer from a solution string and check correctness.

    Returns (predicted_answer, is_correct).
    Uses the grader from baseline-tts when available (math_equal +
    extract_answer), otherwise falls back to a lightweight implementation.
    """
    predicted = extract_answer(solution, data_name)
    if not predicted or predicted.strip() == "":
        return None, False
    correct = is_correct(predicted, ground_truth, data_name)
    return predicted, correct


# ─────────────────────────────────────────────────────────────────────────────
# Single-problem solver
# ─────────────────────────────────────────────────────────────────────────────
def solve_one(
    problem: str,
    ground_truth: Any,
    data_name: str,
    generator: LLMGenerator,
    prm: QwenPRMScorer,
    mcts_steps: int,
    samples_per_action: int,
    exploration_weight: float,
) -> dict:
    """Run StandardMCTS on a single problem; return a result dict."""
    algo  = StandardMCTS(
        samples_per_action=samples_per_action,
        exploration_weight=exploration_weight,
    )
    state = algo.init_tree()

    token_stats = TokenStats()

    generate_fns = {
        "gen_a": make_generate_fn(generator, prm, problem,
                                  temperature=0.7, token_stats=token_stats),
        "gen_b": make_generate_fn(generator, prm, problem,
                                  temperature=1.0, token_stats=token_stats),
    }

    for _ in range(mcts_steps):
        state = algo.step(state, generate_fns)

    pairs = algo.get_state_score_pairs(state)
    if not pairs:
        return {
            "problem":          problem,
            "ground_truth":     ground_truth,
            "best_solution":    None,
            "best_score":       0.0,
            "predicted_answer": None,
            "is_correct":       False,
            "top3":             [],
            "token_stats":      token_stats.to_dict(),
        }

    best_solution, best_score = max(pairs, key=lambda x: x[1])
    predicted, correct = evaluate_answer(best_solution, ground_truth, data_name)

    top3 = top_k(state, algo, k=3)

    return {
        "problem":          problem,
        "ground_truth":     ground_truth,
        "best_solution":    best_solution,
        "best_score":       best_score,
        "predicted_answer": predicted,
        "is_correct":       correct,
        "top3": [{"solution": sol, "score": sc} for sol, sc in top3],
        "token_stats":      token_stats.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="StandardMCTS + PRM on math reasoning benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── dataset ──
    p.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        default="math500",
        help=(
            "Dataset to evaluate.\n"
            + "\n".join(f"  {k}: {v.file}" for k, v in DATASET_REGISTRY.items())
        ),
    )

    # ── model ──
    p.add_argument(
        "--llm",
        metavar="NAME_OR_PATH",
        default="qwen2.5-7b",
        help=(
            "LLM to use. Either a short key or a full HF ID / local path.\n"
            + "Short keys: " + ", ".join(MODEL_REGISTRY.keys())
        ),
    )
    p.add_argument(
        "--llm_path",
        default=None,
        help="Override: explicit local path or HF ID for the LLM "
             "(takes priority over --llm).",
    )
    p.add_argument(
        "--prm_path",
        default=DEFAULT_PRM_PATH,
        help="Path or HF ID for Qwen2.5-Math-PRM-7B.",
    )

    # ── MCTS hyperparams ──
    p.add_argument("--mcts_steps",         type=int,   default=16)
    p.add_argument("--samples_per_action", type=int,   default=2)
    p.add_argument("--exploration_weight", type=float, default=1.414)
    p.add_argument("--max_tokens",         type=int,   default=2048)

    # ── evaluation range ──
    p.add_argument("--num_problems", type=int, default=0,
                   help="Number of problems to run (0 = all).")
    p.add_argument("--start_idx",    type=int, default=0)

    # ── hardware ──
    p.add_argument("--llm_gpu", type=int, default=0)
    p.add_argument("--prm_gpu", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    # ── output ──
    p.add_argument(
        "--output", default=None,
        help="Output JSON file. Defaults to results/<dataset>_<llm>_<timestamp>.json",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Incremental checkpoint writer
# ─────────────────────────────────────────────────────────────────────────────

_TOK_KEYS = [
    "llm_prompt_tokens", "llm_generated_tokens",
    "prm_input_tokens", "total_llm_tokens", "total_tokens",
    "llm_calls", "prm_calls",
]


def _compute_tok_stats(results: list[dict]) -> tuple[dict, dict]:
    n = len(results)
    totals = {k: sum(r["token_stats"].get(k, 0) for r in results) for k in _TOK_KEYS}
    avgs   = {f"avg_{k}": round(totals[k] / n, 1) for k in _TOK_KEYS} if n else {}
    return totals, avgs


def _write_checkpoint(
    out_path: Path,
    results: list[dict],
    n_correct: int,
    total: int,
    base_summary: dict,
    status: str,          # "running" | "completed"
) -> None:
    """
    Atomically rewrite the output JSON with the current results list.
    Uses write-to-tmp + os.replace so a partial read never sees a broken file.
    """
    n           = len(results)
    accuracy    = n_correct / n * 100 if n else 0.0
    tok_totals, tok_avg = _compute_tok_stats(results)

    summary = {
        **base_summary,
        "status":           status,
        "last_updated":     datetime.now().isoformat(),
        "num_completed":    n,
        "num_total":        total,
        "num_correct":      n_correct,
        "running_accuracy": f"{accuracy:.2f}%  ({n_correct}/{n})",
        "token_totals":     tok_totals,
        "token_averages":   tok_avg,
    }

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f,
                  indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)   # atomic on POSIX


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── resolve model path ────────────────────────────────────────────────────
    llm_path = args.llm_path if args.llm_path else resolve_model_path(args.llm)
    llm_tag  = args.llm.replace("/", "_")

    # ── output path ───────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _REPO / "results" / f"{args.dataset}_{llm_tag}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    ds_cfg  = DATASET_REGISTRY[args.dataset]
    samples = load_dataset(ds_cfg, _REPO)

    end_idx = (args.start_idx + args.num_problems
               if args.num_problems > 0 else len(samples))
    subset  = samples[args.start_idx:end_idx]

    logger.info(
        f"Dataset : {args.dataset}  ({len(subset)} problems, "
        f"idx {args.start_idx}–{end_idx-1})"
    )
    logger.info(f"LLM     : {llm_path}")
    logger.info(f"PRM     : {args.prm_path}")
    logger.info(f"Grader  : {'baseline-tts (math_equal)' if _GRADER_AVAILABLE else 'lightweight fallback'}")
    logger.info(f"Output  : {out_path}  (saved after every sample)")

    # Static part of summary — written on every checkpoint
    base_summary = {
        "started_at": datetime.now().isoformat(),
        "dataset":    args.dataset,
        "llm":        llm_path,
        "prm":        args.prm_path,
        "grader":     "baseline-tts/math_equal" if _GRADER_AVAILABLE else "fallback",
        "config": {
            "mcts_steps":         args.mcts_steps,
            "samples_per_action": args.samples_per_action,
            "exploration_weight": args.exploration_weight,
            "max_tokens":         args.max_tokens,
        },
    }

    # ── load models ───────────────────────────────────────────────────────────
    generator = LLMGenerator(
        model_path=llm_path,
        gpu_id=args.llm_gpu,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prm = QwenPRMScorer(
        model_path=args.prm_path,
        gpu_id=args.prm_gpu,
    )

    # ── run MCTS per problem ──────────────────────────────────────────────────
    results   = []
    n_correct = 0

    for i, sample in enumerate(tqdm(subset, desc=f"MCTS/{args.dataset}", unit="prob")):
        problem      = sample[ds_cfg.problem_key]
        ground_truth = sample[ds_cfg.answer_key]
        unique_id    = sample.get("unique_id", sample.get("url", f"idx_{args.start_idx + i}"))

        logger.info(f"[{i+1}/{len(subset)}] {unique_id}")

        try:
            result = solve_one(
                problem=problem,
                ground_truth=ground_truth,
                data_name=ds_cfg.data_name,
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
                "token_stats": TokenStats().to_dict(),
            }

        result["unique_id"] = unique_id

        # Attach running accuracy *to this result* before appending
        n_correct += int(result["is_correct"])
        running_acc = n_correct / (i + 1) * 100
        result["running_accuracy"] = f"{running_acc:.2f}%  ({n_correct}/{i + 1})"

        results.append(result)

        ts = result["token_stats"]
        logger.info(
            f"  pred={result['predicted_answer']}  gt={ground_truth}  "
            f"correct={result['is_correct']}  "
            f"acc={running_acc:.1f}%  "
            f"tok(llm={ts['total_llm_tokens']} prm={ts['prm_input_tokens']} "
            f"total={ts['total_tokens']})"
        )

        # ── incremental save after every sample ──────────────────────────────
        _write_checkpoint(
            out_path, results, n_correct,
            total=len(subset),
            base_summary=base_summary,
            status="running",
        )

    # ── final save ────────────────────────────────────────────────────────────
    _write_checkpoint(
        out_path, results, n_correct,
        total=len(subset),
        base_summary=base_summary,
        status="completed",
    )

    n        = len(results)
    accuracy = n_correct / n * 100 if n else 0.0
    tok_totals, tok_avg = _compute_tok_stats(results)

    logger.info("=" * 64)
    logger.info(f"Dataset  : {args.dataset}  ({n} problems)")
    logger.info(f"Accuracy : {accuracy:.2f}%  ({n_correct}/{n})")
    logger.info(f"Total tokens    : {tok_totals['total_tokens']:,}")
    logger.info(f"  └ LLM prompt  : {tok_totals['llm_prompt_tokens']:,}")
    logger.info(f"  └ LLM gen     : {tok_totals['llm_generated_tokens']:,}")
    logger.info(f"  └ PRM input   : {tok_totals['prm_input_tokens']:,}")
    logger.info(f"Avg per problem : {tok_avg.get('avg_total_tokens', 0):,.1f} tokens")
    logger.info(f"  └ avg LLM     : {tok_avg.get('avg_total_llm_tokens', 0):,.1f}")
    logger.info(f"  └ avg PRM     : {tok_avg.get('avg_prm_input_tokens', 0):,.1f}")
    logger.info(f"  └ avg calls   : "
                f"{tok_avg.get('avg_llm_calls', 0):.1f} LLM  "
                f"{tok_avg.get('avg_prm_calls', 0):.1f} PRM")
    logger.info(f"Saved to : {out_path}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
