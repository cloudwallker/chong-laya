"""Exercise form state, error presentation, and result rendering without network."""
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from chong.decision import DecisionEngine


class FixedAgent:
    def predict(self, state, questions):
        return {"model": "laya-multilingual", "answers": {"decision": {
            "type": "choice", "choice": "skip", "confidence": 0.73,
            "probabilities": {"buy": 0.1, "wait": 0.2, "skip": 0.7}}},
            "usage": {"input_tokens": 60, "output_tokens": 0}}


class AppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        return app

    def test_empty_input_shows_message_without_loading_model(self):
        with patch("chong.decision._load_laya", side_effect=AssertionError("must not load")):
            app = self.app()
            app.text_area(key="context").set_value("")
            app.button(key="decide").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("输入" in warning.value for warning in app.warning))

    def test_result_is_actual_model_choice_and_editing_clears_stale_result(self):
        with patch("chong.decision._load_laya", return_value=FixedAgent()):
            app = self.app()
            app.text_area(key="context").set_value("已有能用的键盘，预算紧，但很想趁促销买新的。")
            app.button(key="decide").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["result"]["choice"], "skip")
            self.assertTrue(any("未经校准" in caption.value for caption in app.caption))
            app.text_area(key="context").set_value("我想买一盏灯，尚未确认预算。").run()
            self.assertIsNone(app.session_state["result"])

    def test_model_load_failure_is_visible_and_keeps_input(self):
        with patch("chong.decision._load_laya", side_effect=OSError("offline")):
            app = self.app()
            app.text_area(key="context").set_value("旧饭盒坏了，预算充足，想买新饭盒。")
            app.button(key="decide").click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(any("加载失败" in error.value for error in app.error))
        self.assertIn("饭盒", app.text_area(key="context").value)


if __name__ == "__main__":
    unittest.main()
