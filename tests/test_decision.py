import math
import os
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from chong.decision import (
    DEFAULT_ORDER,
    LABELS,
    MODEL_ID,
    DecisionEngine,
    InferenceError,
    InputError,
    ModelLoadError,
    decide,
)
from chong import decision as decision_module


class FakeAgent:
    def __init__(self, answer=None):
        self.answer = answer or {
            "type": "choice",
            "choice": "wait",
            "probabilities": {"buy": 0.2, "wait": 0.6, "skip": 0.2},
            "confidence": 0.5,
        }
        self.requests = []

    def predict(self, state, questions):
        self.requests.append((state, questions))
        return {"model": "laya-rl-agent", "answers": {"decision": self.answer},
                "usage": {"input_tokens": 25, "output_tokens": 0}}


class DecisionTests(unittest.TestCase):
    def test_token_budget_rejects_state_before_sdk_can_truncate_it(self):
        class CharacterTokenizer:
            mask_token = "[MASK]"

            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": list(range(len(text)))}

        agent = FakeAgent()
        agent.tok = CharacterTokenizer()
        agent.cfg = {"max_len": 20, "head_max_len": 10}
        engine = DecisionEngine(loader=lambda *_args, **_kw: agent)
        with self.assertRaisesRegex(InputError, "输入过长"):
            engine.decide("需要新耳机预算不足。")  # 10 字，状态预算仅 6 token。
        self.assertEqual(agent.requests, [])

    def test_token_preflight_replaces_mask_token_like_sdk(self):
        seen = []

        class CharacterTokenizer:
            mask_token = "[MASK]"

            def __call__(self, text, *, add_special_tokens):
                seen.append((text, add_special_tokens))
                return {"input_ids": list(range(len(text)))}

        agent = FakeAgent()
        agent.tok = CharacterTokenizer()
        agent.cfg = {"max_len": 100, "head_max_len": 20}
        engine = DecisionEngine(loader=lambda *_args, **_kw: agent)
        engine.decide("预算[MASK]不足。")
        self.assertEqual(seen, [("预算 不足。", False)])
        self.assertEqual(agent.requests[0][0], "预算[MASK]不足。")

    def test_default_loader_configures_runtime_before_sdk_load(self):
        observed = []

        def load(model, *, device):
            observed.append((model, device, os.environ.get("HF_HOME"),
                             os.environ.get("HF_HUB_DISABLE_TELEMETRY")))
            return FakeAgent()

        fake_laya = types.SimpleNamespace(load=load)
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"laya": fake_laya}):
            decision_module._load_laya(MODEL_ID, device="cpu")
        self.assertEqual(observed[0][:2], (MODEL_ID, "cpu"))
        self.assertIsNotNone(observed[0][2])
        self.assertTrue(observed[0][2].endswith("huggingface"))
        self.assertEqual(observed[0][3], "1")

    def test_prompt_keeps_promotion_and_existing_item_as_joint_skip_reason(self):
        result = decide("旧耳机够用，折扣看着很诱人。",
                        engine=DecisionEngine(loader=lambda *_args, **_kw: FakeAgent()))
        question = result.request["questions"]["decision"]
        self.assertIn("现有物品足够且主要受促销驱动", question["criteria"]["skip"])
        self.assertIn("现有物品足够且主要受促销驱动", question["instructions"])
        self.assertIn("预算有压力", question["criteria"]["skip"])

    def test_prediction_preserves_sdk_choice_and_auditable_payload(self):
        agent = FakeAgent({"type": "choice", "choice": "skip",
                           "probabilities": {"buy": 0.6, "wait": 0.3, "skip": 0.1}})
        loads = []

        def loader(model, *, device):
            loads.append((model, device))
            return agent

        engine = DecisionEngine(loader=loader)
        self.assertEqual(loads, [])
        result = decide("  月末预算紧张，但想买新耳机。  ", engine=engine,
                        option_order=("skip", "wait", "buy"))
        self.assertEqual(loads, [(MODEL_ID, "cpu")])
        self.assertEqual(result.choice, "skip")
        self.assertEqual(result.probabilities,
                         {"buy": 0.6, "wait": 0.3, "skip": 0.1})
        self.assertEqual(result.model, MODEL_ID)
        self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertEqual(result.request["state"], "月末预算紧张，但想买新耳机。")
        self.assertEqual(list(result.request["questions"]["decision"]["criteria"]),
                         ["skip", "wait", "buy"])
        self.assertEqual(agent.requests[0],
                         (result.request["state"], result.request["questions"]))
        self.assertEqual(result.raw_response["answers"]["decision"]["choice"], "skip")
        self.assertEqual(result.to_dict()["choice"], "skip")
        self.assertEqual(set(result.to_dict()), {"choice", "probabilities", "elapsed_ms",
                                                  "model", "request", "raw_response"})

    def test_default_order_and_labels(self):
        agent = FakeAgent()
        result = decide("确实需要，预算充足。", engine=DecisionEngine(loader=lambda *_args, **_kw: agent))
        self.assertEqual(DEFAULT_ORDER, ("buy", "wait", "skip"))
        self.assertEqual(LABELS, {"buy": "冲", "wait": "等等", "skip": "不冲"})
        self.assertEqual(list(result.request["questions"]["decision"]["criteria"]),
                         ["buy", "wait", "skip"])
        self.assertEqual(result.request["questions"]["decision"]["type"], "choice")

    def test_rejects_invalid_input_before_loading(self):
        calls = []
        engine = DecisionEngine(loader=lambda *_args, **_kw: calls.append(True))
        for value in (None, "", "   \n", 123, "需" * 2001):
            with self.subTest(value=str(value)[:12]), self.assertRaisesRegex(InputError, "输入"):
                engine.decide(value)
        self.assertEqual(calls, [])

    def test_rejects_invalid_option_order_before_loading(self):
        engine = DecisionEngine(loader=lambda *_args, **_kw: self.fail("不应加载"))
        for order in (("buy", "wait"), ("buy", "buy", "skip"),
                      ("buy", "wait", "other"), "buy,wait,skip"):
            with self.subTest(order=order), self.assertRaises(InputError):
                engine.decide("需要耳机", option_order=order)

    def test_load_failure_is_explicit(self):
        def broken(*_args, **_kwargs):
            raise OSError("missing weights")

        engine = DecisionEngine(loader=broken)
        with self.assertRaises(ModelLoadError):
            engine.decide("需要耳机")

    def test_inference_failure_is_explicit(self):
        class BrokenAgent:
            def predict(self, *_args):
                raise RuntimeError("backend failure")

        engine = DecisionEngine(loader=lambda *_args, **_kw: BrokenAgent())
        with self.assertRaises(InferenceError):
            engine.decide("需要耳机")

    def test_rejects_invalid_sdk_choices_and_probabilities(self):
        invalid_answers = [
            {"choice": "unknown", "probabilities": {"buy": 0.2, "wait": 0.6, "skip": 0.2}},
            {"choice": "buy", "probabilities": {"buy": 0.5, "wait": 0.5}},
            {"choice": "buy", "probabilities": {"buy": math.nan, "wait": 0.5, "skip": 0.5}},
            {"choice": "buy", "probabilities": {"buy": -0.1, "wait": 0.5, "skip": 0.6}},
            {"choice": "buy", "probabilities": {"buy": 1.0005, "wait": 0, "skip": 0}},
            {"choice": "buy", "probabilities": {"buy": 0.2, "wait": 0.2, "skip": 0.2}},
        ]
        for answer in invalid_answers:
            with self.subTest(answer=answer), self.assertRaises(InferenceError):
                DecisionEngine(loader=lambda *_args, **_kw: FakeAgent(answer)).decide("需要耳机")

    def test_reuses_one_loaded_agent_and_serializes_prediction(self):
        active = 0
        peak = 0
        loads = 0
        guard = threading.Lock()

        class SlowAgent(FakeAgent):
            def predict(self, state, questions):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.01)
                try:
                    return super().predict(state, questions)
                finally:
                    with guard:
                        active -= 1

        agent = SlowAgent()

        def loader(*_args, **_kwargs):
            nonlocal loads
            time.sleep(0.01)
            loads += 1
            return agent

        engine = DecisionEngine(loader=loader)
        results = []
        threads = [threading.Thread(target=lambda: results.append(engine.decide("需要耳机")))
                   for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(results), 5)
        self.assertEqual(loads, 1)
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
