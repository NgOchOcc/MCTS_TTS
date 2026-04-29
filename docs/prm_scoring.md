# Thay LLM Judging bằng PRM Scoring trong TreeQuest

## 1. Cách TreeQuest dùng LLM để judge/score hiện tại

### 1.1 Kiến trúc tổng quan

TreeQuest **không gọi LLM trực tiếp** — framework được thiết kế agnostic với nguồn sinh score. Toàn bộ logic LLM judging nằm trong hàm `generate_fn` do người dùng tự viết.

**Luồng dữ liệu:**

```
MCTS select node
    → ask_batch() / step() trả ra Trial
    → user gọi generate_fn(parent_state) → (new_state, score ∈ [0,1])
    → tell(trial_id, result) nạp score vào cây
    → backpropagate cập nhật value_sums / visit_counts
```

### 1.2 Interface generate_fn

```python
# Định nghĩa trong src/treequest/types.py
GenerateFnType = Callable[[Optional[NodeStateT]], Tuple[NodeStateT, float]]
```

Hàm nhận `parent_state` (có thể `None` nếu là root), trả ra `(new_state, score)`.

**Ví dụ pattern LLM judging điển hình:**

```python
def generate_with_llm_judge(parent_state: str | None) -> tuple[str, float]:
    # Bước 1: Generate solution từ LLM
    if parent_state is None:
        solution = llm.generate(problem_prompt)
    else:
        solution = llm.refine(problem_prompt, parent_state)

    # Bước 2: Dùng LLM thứ 2 (judge) để score solution
    judge_prompt = f"""
    Problem: {problem_prompt}
    Solution: {solution}
    Rate this solution from 0.0 to 1.0. Return only the number.
    """
    raw_score = float(judge_llm.generate(judge_prompt))
    score = max(0.0, min(1.0, raw_score))   # clamp vào [0, 1]

    return solution, score
```

### 1.3 Điểm tiêm score trong StandardMCTS

**`standard_mcts.py` — phương thức `tell()`:**

```python
def tell(self, state, trial_id, result):
    new_state, new_score = result          # score đến từ đây
    new_node = state.tree.add_node(
        (new_state, new_score),            # gán vào Node.score
        parent_node,
        trial_id=trial_id
    )
    state = self._backpropagate(state, new_node, new_score)
    return state
```

**`_backpropagate()`** — lan truyền score lên gốc:

```python
def _backpropagate(self, state, node, score):
    current = node
    while current is not None:
        state.visit_counts[current.id] += 1
        state.value_sums[current.id] += score   # tích lũy để tính avg
        current = current.parent
    return state
```

**UCT selection** dùng `value_sum / visit_count` làm exploitation term:

```python
def _uct_score(self, state, node, parent):
    avg_value = state.value_sums[node.id] / state.visit_counts[node.id]
    exploration = (
        self.exploration_weight
        * state.priors[node.id]
        * sqrt(log(parent_visits) / node_visits)
    )
    return avg_value + exploration
```

---

## 2. PRM là gì và tại sao phù hợp hơn LLM judging

| Tiêu chí | LLM Judge | PRM (Process Reward Model) |
|---|---|---|
| Latency | Cao (full generation) | Thấp (forward pass) |
| Consistency | Không ổn định, prompt-sensitive | Deterministic |
| Granularity | Thường chỉ outcome | Có thể score từng bước (step-level) |
| Cost | Đắt (large model) | Rẻ (fine-tuned smaller model) |
| Domain | General | Specialized theo task |

PRM được fine-tune để predict xác suất mỗi **bước lý luận trung gian** là đúng, không chỉ đánh kết quả cuối. Điều này ăn khớp tự nhiên với cây MCTS — mỗi node là một bước tư duy.

---

## 3. Thiết kế thay LLM bằng PRM

### 3.1 Sơ đồ thay thế

```
Trước (LLM judging):
  generate_fn(parent_state)
      → LLM.generate(prompt)  →  solution_text
      → judge_LLM.score(solution_text)  →  float score
      → return (solution_text, score)

Sau (PRM scoring):
  generate_fn(parent_state)
      → LLM.generate(prompt)  →  solution_text  (vẫn dùng LLM để gen)
      → PRM.score(problem, solution_text)  →  float score
      → return (solution_text, score)
```

Chỉ thay phần **scoring** — phần generation giữ nguyên.

### 3.2 Interface PRM cần implement

```python
from abc import ABC, abstractmethod

class PRMScorer(ABC):
    @abstractmethod
    def score(
        self,
        problem: str,
        solution_steps: list[str],   # danh sách bước lý luận
    ) -> float:
        """
        Trả về scalar ∈ [0, 1] đại diện cho xác suất solution đúng.
        Có thể là:
          - min score qua các bước (pessimistic)
          - product score qua các bước
          - score bước cuối cùng
        """
        ...
```

### 3.3 Ví dụ implement với Hugging Face PRM

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HuggingFacePRM(PRMScorer):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.device = device

    def score(self, problem: str, solution_steps: list[str]) -> float:
        step_scores = []
        context = problem

        for step in solution_steps:
            inputs = self.tokenizer(
                context,
                step,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                prob = torch.sigmoid(logits[0, 1]).item()  # prob correct
            
            step_scores.append(prob)
            context = context + "\n" + step   # tích lũy context

        if not step_scores:
            return 0.0
        
        # Dùng min để pessimistic: nếu 1 bước sai thì cả chain sai
        return min(step_scores)
```

### 3.4 Integrate vào generate_fn của TreeQuest

```python
from treequest import StandardMCTS

# Khởi tạo PRM một lần
prm = HuggingFacePRM("Qwen/Qwen2.5-Math-PRM-7B")

def make_generate_fn_with_prm(
    llm,
    problem: str,
    prm_scorer: PRMScorer,
):
    def generate_fn(parent_state: str | None) -> tuple[str, float]:
        # --- Generation: vẫn dùng LLM ---
        if parent_state is None:
            raw_output = llm.generate(
                f"Solve step by step:\n{problem}"
            )
        else:
            raw_output = llm.generate(
                f"Continue or improve this solution:\n{parent_state}"
            )

        # --- Parsing steps ---
        # PRM cần list các bước; tách theo dấu hiệu bước
        steps = parse_steps(raw_output)   # xem hàm helper bên dưới

        # --- Scoring: thay bằng PRM ---
        score = prm_scorer.score(problem, steps)

        return raw_output, score

    return generate_fn


def parse_steps(text: str) -> list[str]:
    """Tách solution text thành list các bước."""
    # Cách đơn giản: tách theo "Step N:" hoặc dòng trống
    import re
    steps = re.split(r"(?:Step \d+:|\n\n)", text)
    return [s.strip() for s in steps if s.strip()]


# Usage
algo = StandardMCTS(samples_per_action=2, exploration_weight=1.414)
tree_state = algo.init_tree()

generate_fns = {
    "llm_a": make_generate_fn_with_prm(llm_model_a, problem, prm),
    "llm_b": make_generate_fn_with_prm(llm_model_b, problem, prm),
}

for _ in range(100):
    tree_state = algo.step(tree_state, generate_fns)
```

---

## 4. Chiến lược tổng hợp score từ các bước

PRM trả ra score per-step. Có nhiều cách aggregate:

```python
def aggregate_step_scores(step_scores: list[float], strategy: str) -> float:
    if not step_scores:
        return 0.0
    
    if strategy == "min":
        # Pessimistic: chain chỉ tốt bằng bước yếu nhất
        return min(step_scores)
    
    elif strategy == "product":
        # Xác suất tất cả bước đều đúng (giả định independence)
        result = 1.0
        for s in step_scores:
            result *= s
        return result
    
    elif strategy == "last":
        # Chỉ quan tâm kết quả cuối
        return step_scores[-1]
    
    elif strategy == "mean":
        return sum(step_scores) / len(step_scores)
    
    elif strategy == "weighted_last":
        # Trọng số tăng dần, bước cuối quan trọng nhất
        weights = [i + 1 for i in range(len(step_scores))]
        return sum(w * s for w, s in zip(weights, step_scores)) / sum(weights)
```

**Khuyến nghị:** Dùng `"min"` hoặc `"product"` vì MCTS cần phân biệt chuỗi lý luận tốt/xấu rõ ràng — nếu aggregate bằng mean, các node có 1 bước sai vẫn có score cao và gây nhiễu selection.

---

## 5. Các PRM model có thể dùng ngay

| Model | Backbone | Domain | Link |
|---|---|---|---|
| `Qwen/Qwen2.5-Math-PRM-7B` | Qwen2.5 7B | Math | Hugging Face |
| `peiyi9979/math-shepherd-mistral-7b-prm` | Mistral 7B | Math | Hugging Face |
| `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | LLaMA 3.1 8B | Reasoning | Hugging Face |
| `ScalableMath/Llemma-7b-prm-prm800k-level-1to3` | Llemma 7B | Math | Hugging Face |

Với domain khác (code, science, ...) cần fine-tune PRM riêng trên dữ liệu có process-level annotation.

---

## 6. Lưu ý khi tích hợp

### 6.1 Score phải ở trong `[0, 1]`
TreeQuest validate cứng tại `tree.add_node()`. Đảm bảo PRM output được clamp:

```python
score = max(0.0, min(1.0, raw_prm_score))
```

### 6.2 Batch inference để tăng throughput
Khi dùng `ask_batch()`, nhiều trial được tạo ra cùng lúc — nên batch PRM calls:

```python
state, trials = algo.ask_batch(state, batch_size=8, actions=["a", "b"])

# Generate tất cả solutions trước
solutions = [generate_solution(t.parent_state) for t in trials]

# Batch score với PRM
scores = prm.score_batch(problem, solutions)   # implement batch method

for trial, solution, score in zip(trials, solutions, scores):
    state = algo.tell(state, trial.trial_id, (solution, score))
```

### 6.3 Caching PRM scores
Nếu nhiều node có cùng partial solution text, cache score để tránh redundant forward pass:

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_prm_score(problem: str, solution: str) -> float:
    steps = parse_steps(solution)
    return prm.score(problem, steps)
```

### 6.4 PRM score cho step-level MCTS
Nếu muốn mỗi **node = một bước lý luận** (finer granularity):

```python
def generate_one_step(parent_state: str | None) -> tuple[str, float]:
    # Chỉ generate 1 bước tiếp theo
    if parent_state is None:
        step = llm.generate_first_step(problem)
        full_chain = step
    else:
        step = llm.generate_next_step(problem, parent_state)
        full_chain = parent_state + "\n" + step

    # Score chỉ bước này trong context của chain đầy đủ
    all_steps = parse_steps(full_chain)
    score = prm.score(problem, all_steps)[-1]  # score bước mới nhất

    return full_chain, score
```

Cách này khai thác triệt để MCTS — mỗi node tương ứng đúng 1 reasoning step, và PRM cho biết bước đó có hứa hẹn không.
