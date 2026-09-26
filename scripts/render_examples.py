"""Render README figures from recorded inference, without loading model weights.

Run from the repository root: python scripts/render_examples.py
Requires the plotting dependencies in requirements-dev.txt.
"""
import json
import os
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

BG = "#F8F7F3"
INK = "#252924"
MUTED = "#737C70"
ORANGE = "#E7683C"
GREEN = "#42775E"
LINE = "#DDDCD2"
LABELS = {"buy": "冲 / BUY", "wait": "等等 / WAIT", "skip": "不冲 / SKIP"}


def configure_fonts():
    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei"]
    selected = next((name for name in candidates if name in available), None)
    if selected is None:
        raise SystemExit("Install Microsoft YaHei or Noto Sans CJK SC to render the Chinese figure labels.")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False,
                         "text.color": INK, "font.size": 12})


def text(ax, x, y, value, size=14, color=INK, weight="normal", **kwargs):
    ax.text(x, y, value, transform=ax.transAxes, fontsize=size, color=color,
            fontweight=weight, va="top", **kwargs)


def box(ax, x, y, width, height, face="white"):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1,
                 edgecolor=LINE, facecolor=face, transform=ax.transAxes))


def render_demo(report, output):
    row = next(item for item in report["rows"] if item["id"] == "buy-07")
    result = row["baseline"]
    fig = plt.figure(figsize=(13.6, 7.9), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    text(ax, .06, .945, "●  CHONG / DECISION LAB", 12, ORANGE, "bold")
    text(ax, .06, .88, "今天你冲了吗？", 34, weight="bold")
    text(ax, .06, .785, "一个情境，三个选项，一次真实的模型选择。", 16, MUTED)
    text(ax, .06, .733, "Recorded inference visualization · Not an application screenshot", 10, MUTED)
    box(ax, .06, .23, .405, .43)
    box(ax, .535, .23, .405, .43)
    text(ax, .085, .63, "01 / STATE", 11, MUTED, "bold")
    text(ax, .085, .573, "这次想买什么？", 19, weight="bold")
    phrases = re.findall(r"[^，；。]+[，；。]?", row["context"])
    wrapped = "\n".join(line for phrase in phrases for line in textwrap.wrap(phrase, width=22))
    text(ax, .085, .49, wrapped, 14, linespacing=1.6)
    text(ax, .085, .292, f"样例 {row['id']}  ·  预期标签：{LABELS[row['expected']]}", 10, MUTED)
    text(ax, .488, .467, "→", 24, MUTED, ha="center")
    text(ax, .56, .63, "02 / CHOICE", 11, MUTED, "bold")
    text(ax, .56, .572, LABELS[result["choice"]], 27, GREEN, "bold")
    text(ax, .905, .566, f"{result['elapsed_ms']:.1f} ms", 12, MUTED, ha="right")
    for index, key in enumerate(LABELS):
        y = .462 - index * .061
        value = result["probabilities"][key]
        text(ax, .56, y, LABELS[key], 11)
        ax.plot([.665, .86], [y - .016] * 2, color="#EEECE5", lw=9,
                solid_capstyle="round", transform=ax.transAxes)
        if value:
            ax.plot([.665, .665 + .195 * value], [y - .016] * 2,
                    color=GREEN if key == result["choice"] else "#A9B2A4", lw=9,
                    solid_capstyle="round", transform=ax.transAxes)
        text(ax, .91, y, f"{value:.1%}", 11, ha="right")
    text(ax, .56, .282, "未经校准 · 不代表“买对的概率”", 11, MUTED)
    text(ax, .06, .155, "LAYA MULTILINGUAL  /  CPU  /  CHOICE ONLY", 11, MUTED, "bold")
    text(ax, .06, .105, "数据：reports/baseline/results.json · 完整评测包含成功和失败案例", 10, MUTED)
    text(ax, .94, .105, "@cloudwallker", 11, MUTED, ha="right")
    fig.savefig(output, dpi=150, facecolor=BG)
    plt.close(fig)


def render_benchmark(report, output):
    summary = report["summary"]
    fig = plt.figure(figsize=(13.6, 7.9), dpi=150, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    text(ax, .06, .945, "CHONG / BASELINE EVALUATION", 12, ORANGE, "bold")
    text(ax, .06, .88, "把结果如实摆出来。", 32, weight="bold")
    text(ax, .06, .78, "30 条中文案例 · 15 组等价改写 · 真实本地推理", 15, MUTED)
    for x, heading, value, detail in (
        (.06, "标签一致 / LABEL AGREEMENT", f"{summary['matched']} / {summary['total']}", "按预设标准标注"),
        (.36, "改写一致 / PARAPHRASE", f"{summary['paraphrase']['consistent']} / {summary['paraphrase']['comparable']}", "一致不等于正确"),
        (.66, "逆序一致 / OPTION ORDER", f"{summary['option_order']['consistent']} / {summary['option_order']['comparable']}", "只改变选项排列"),
    ):
        box(ax, x, .48, .27, .23)
        text(ax, x + .017, .684, heading, 9, MUTED, "bold")
        text(ax, x + .017, .627, value, 32, weight="bold")
        text(ax, x + .017, .535, detail, 10, MUTED)
    chart = fig.add_axes([.15, .17, .40, .245], facecolor=BG)
    keys = list(LABELS)
    values = [summary["per_label"][key]["matched"] for key in keys]
    chart.barh(range(3), [summary["per_label"][key]["total"] for key in keys], height=.45, color="#E8E6DD")
    chart.barh(range(3), values, height=.45, color=[GREEN, "#BE944A", ORANGE])
    chart.set_yticks(range(3), [LABELS[key] for key in keys], fontsize=11)
    chart.invert_yaxis()
    chart.set_xlim(0, max(summary["per_label"][key]["total"] for key in keys) + 2)
    chart.set_xticks([])
    chart.tick_params(axis="y", length=0, pad=15)
    for spine in chart.spines.values():
        spine.set_visible(False)
    for index, key in enumerate(keys):
        item = summary["per_label"][key]
        chart.text(item["total"] + .2, index, f"{item['matched']}/{item['total']}", va="center", color=INK, fontsize=12)
    text(ax, .66, .408, "CPU 预热后中位耗时", 12, MUTED)
    text(ax, .66, .35, f"{summary['latency_ms']['median']:.1f} ms", 27, weight="bold")
    text(ax, .66, .265, "未微调 · 未校准\n完整保留所有错误案例", 12, MUTED, linespacing=1.8)
    date = report["created_at"].split("T")[0]
    text(ax, .06, .095, f"{date}  /  Laya 0.3.20  /  Windows CPU  /  本地小样本结果，不代表通用性能", 10, MUTED)
    fig.savefig(output, dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    configure_fonts()
    report = json.loads((ROOT / "reports/baseline/results.json").read_text(encoding="utf-8"))
    directory = ROOT / "docs/images"
    directory.mkdir(parents=True, exist_ok=True)
    render_demo(report, directory / "demo.png")
    render_benchmark(report, directory / "benchmark.png")
    print("Rendered docs/images/demo.png and docs/images/benchmark.png from the baseline report.")


if __name__ == "__main__":
    main()
