"""用官方 Laya choice 接口生成可检查的单轮购物决策。"""

from dataclasses import asdict, dataclass
import math
from numbers import Real
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple


MODEL_ID = "convaiinnovations/laya-multilingual"
LABELS = {"buy": "冲", "wait": "等等", "skip": "不冲"}
DEFAULT_ORDER = ("buy", "wait", "skip")
MAX_CONTEXT_CHARS = 500


class InputError(ValueError):
    """用户输入或选项顺序不合法。"""


class ModelLoadError(RuntimeError):
    """官方 Laya 模型无法初始化。"""


class InferenceError(RuntimeError):
    """模型推理失败或返回格式不合法。"""


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    probabilities: Dict[str, float]
    elapsed_ms: float
    model: str
    request: Dict[str, Any]
    raw_response: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """返回可序列化的独立字典，供页面与评测记录使用。"""
        return asdict(self)


def _load_laya(model_id: str, *, device: str) -> Any:
    from .runtime import configure_runtime

    configure_runtime()
    import laya  # 仅在第一次真正推理时导入 SDK。

    return laya.load(model_id, device=device)


def _validate_context(context: str) -> str:
    if not isinstance(context, str) or not context.strip():
        raise InputError("请输入购物情况，不能留空。")
    context = context.strip()
    if len(context) > MAX_CONTEXT_CHARS:
        raise InputError(f"输入过长，请控制在 {MAX_CONTEXT_CHARS} 字以内。")
    return context


def _validate_order(option_order: Optional[Tuple[str, str, str]]) -> Tuple[str, str, str]:
    if option_order is None:
        return DEFAULT_ORDER
    if (not isinstance(option_order, (tuple, list)) or len(option_order) != 3
            or not all(isinstance(key, str) for key in option_order)
            or set(option_order) != set(DEFAULT_ORDER)):
        raise InputError("选项顺序必须恰好包含 buy、wait、skip，各一次。")
    return tuple(option_order)


def _build_questions(order: Tuple[str, str, str]) -> Dict[str, Any]:
    criteria = {
        "buy": "冲：用途明确、预算足够、现有物品不能满足",
        "wait": "等等：其余情况",
        "skip": "不冲：预算有压力；或现有物品足够且主要受促销驱动",
    }
    return {"decision": {
        "type": "choice",
        "instructions": "按购物情况选一项：预算有压力，或现有物品足够且主要受促销驱动，选不冲；用途明确、预算足够且无替代选冲；其他选等等。",
        "criteria": {key: criteria[key] for key in order},
    }}


def _validate_token_budget(agent: Any, state: str) -> None:
    """用 SDK 同一 tokenizer 拦截会被默认上下文预算截断的状态。"""
    tok = getattr(agent, "tok", None)
    cfg = getattr(agent, "cfg", None)
    if tok is None or not isinstance(cfg, dict):
        return  # 兼容仅实现 predict 的外部 Agent。

    try:
        max_len = cfg.get("max_len", 512)
        head_max_len = cfg.get("head_max_len", 192)
        budget = max_len - head_max_len - 4
        token_ids = tok(state.replace(tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"]
    except Exception as exc:
        raise InferenceError("Laya 输入分词检查失败。") from exc
    if len(token_ids) > budget:
        raise InputError(f"输入过长：模型最多可读取约 {budget} 个状态 token，请缩短描述。")


def _validate_response(response: Any) -> Tuple[str, Dict[str, float]]:
    try:
        answer = response["answers"]["decision"]
        choice = answer["choice"]
        probabilities = answer["probabilities"]
    except (TypeError, KeyError) as exc:
        raise InferenceError("模型响应缺少 choice 或 probabilities。") from exc

    if choice not in DEFAULT_ORDER:
        raise InferenceError("模型返回了不合法的选择。")
    if not isinstance(probabilities, dict) or set(probabilities) != set(DEFAULT_ORDER):
        raise InferenceError("模型返回的概率选项不完整。")
    if not all(isinstance(value, Real) and not isinstance(value, bool)
               and math.isfinite(value) and 0 <= value <= 1
               for value in probabilities.values()):
        raise InferenceError("模型返回的概率不是有限的 0 到 1 之间的数。")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=0.001):
        raise InferenceError("模型返回的概率总和不接近 1。")
    return choice, dict(probabilities)


class DecisionEngine:
    """懒加载一次模型，并串行使用同一 Agent 推理。"""

    def __init__(self, device: str = "cpu", *, loader: Optional[Callable[..., Any]] = None):
        self.device = device
        self._loader = loader if loader is not None else _load_laya
        self._agent = None
        self._lock = threading.Lock()

    def decide(self, context: str, *, option_order=None) -> DecisionResult:
        state = _validate_context(context)
        order = _validate_order(option_order)
        questions = _build_questions(order)
        request = {"state": state, "questions": questions}

        with self._lock:
            if self._agent is None:
                try:
                    agent = self._loader(MODEL_ID, device=self.device)
                    if not callable(getattr(agent, "predict", None)):
                        raise TypeError("loader did not return a Laya Agent")
                except Exception as exc:
                    raise ModelLoadError("Laya 模型加载失败，请检查依赖、缓存或网络。") from exc
                self._agent = agent

            _validate_token_budget(self._agent, state)
            started = time.perf_counter()
            try:
                response = self._agent.predict(state, questions)
            except Exception as exc:
                raise InferenceError("Laya 模型推理失败。") from exc
            elapsed_ms = (time.perf_counter() - started) * 1000

        choice, probabilities = _validate_response(response)
        return DecisionResult(choice, probabilities, elapsed_ms, MODEL_ID, request, response)


_default_engine = DecisionEngine()


def decide(context: str, *, engine: Optional[DecisionEngine] = None,
           option_order=None) -> DecisionResult:
    """使用默认或指定引擎作一次模型决策。"""
    return (engine if engine is not None else _default_engine).decide(
        context, option_order=option_order)
