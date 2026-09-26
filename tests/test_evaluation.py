"""Catch misleading metrics, lost failures, and accidental inclusion of warmup."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chong.evaluation import load_cases, run_evaluation, write_report, runtime_metadata


CASES = [
    {"id": "b1", "context": "b1", "expected": "buy", "category": "need", "pair_id": "b", "variant": "original"},
    {"id": "b2", "context": "b2", "expected": "buy", "category": "need", "pair_id": "b", "variant": "paraphrase"},
    {"id": "w1", "context": "w1", "expected": "wait", "category": "missing", "pair_id": "w", "variant": "original"},
    {"id": "w2", "context": "w2", "expected": "wait", "category": "missing", "pair_id": "w", "variant": "paraphrase"},
]


class Result:
    def __init__(self, choice, elapsed):
        self.choice = choice
        self.elapsed_ms = elapsed

    def to_dict(self):
        return {"choice": self.choice, "elapsed_ms": self.elapsed_ms,
                "model": "test-double", "probabilities": {"buy": 0.3, "wait": 0.3, "skip": 0.4},
                "request": {}, "raw_response": {"fixture": True}}


class ScriptedEngine:
    device = "test"

    def __init__(self, fail=None):
        self.calls = 0
        self.fail = fail

    def decide(self, context, *, option_order=None):
        self.calls += 1
        if context == self.fail:
            raise RuntimeError("fixture failure")
        choices = {"b1": "buy", "b2": "buy", "w1": "wait", "w2": "skip"}
        if option_order and context == "b1":
            choices[context] = "wait"
        # The warmup is intentionally much slower than all measured calls.
        return Result(choices[context], 9999 if self.calls == 1 else 10)


class EvaluationTests(unittest.TestCase):
    def test_metadata_reads_actual_sdk_cache_not_project_default(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            ref = cache / "models--convaiinnovations--laya-multilingual/refs/main"
            ref.parent.mkdir(parents=True)
            ref.write_text("a" * 40, encoding="utf-8")
            with patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)):
                self.assertEqual(runtime_metadata()["checkpoint_revision"], "a" * 40)

    def test_metrics_count_wrong_predictions_and_exclude_warmup(self):
        report = run_evaluation(CASES, ScriptedEngine())
        summary = report["summary"]
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["matched"], 3)
        self.assertEqual(summary["label_agreement"], 0.75)
        self.assertEqual(summary["paraphrase"], {"consistent": 1, "comparable": 2, "total_pairs": 2, "agreement": 0.5})
        self.assertEqual(summary["option_order"]["agreement"], 0.75)
        self.assertEqual(summary["latency_ms"]["median"], 10)
        self.assertEqual(len(report["rows"]), 4)
        self.assertEqual(report["rows"][3]["expected"], "wait")
        self.assertEqual(report["rows"][3]["baseline"]["choice"], "skip")

    def test_errors_remain_visible_in_denominators_and_rows(self):
        report = run_evaluation(CASES, ScriptedEngine(fail="b2"))
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["completed"], 3)
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["summary"]["label_agreement"], 0.5)
        self.assertEqual(report["summary"]["option_order"]["comparable"], 3)
        self.assertEqual(report["summary"]["paraphrase"]["comparable"], 1)
        self.assertIn("fixture failure", report["rows"][1]["error"])

    def test_report_roundtrip_preserves_wrong_cases_and_raw_result(self):
        report = run_evaluation(CASES, ScriptedEngine())
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, Path(directory))
            saved = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
            markdown = (Path(directory) / "README.md").read_text(encoding="utf-8")
        self.assertEqual(saved["rows"][3]["baseline"]["raw_response"], {"fixture": True})
        self.assertIn("w2", markdown)
        self.assertIn("75.0%", markdown)
        self.assertIn("未经校准", markdown)

    def test_case_loader_rejects_duplicate_ids_and_mismatched_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            for invalid in ([CASES[0], CASES[0]], [CASES[0], dict(CASES[1], expected="skip")]):
                path.write_text("\n".join(json.dumps(x) for x in invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_cases(path)

    def test_no_cases_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            run_evaluation([], ScriptedEngine())


if __name__ == "__main__":
    unittest.main()
