"""Small, reproducible evaluation. Labels are rubric agreement, not purchase truth."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import statistics

LABELS = {"buy": "冲", "wait": "等等", "skip": "不冲"}
REVERSED_ORDER = ("skip", "wait", "buy")


def load_cases(path):
    cases = []
    ids = set()
    pairs = defaultdict(list)
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for field in ("id", "context", "expected", "category", "pair_id", "variant"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"第 {line_number} 行缺少有效字段 {field}")
        if case["expected"] not in LABELS or case["id"] in ids:
            raise ValueError(f"第 {line_number} 行标签不合法或 ID 重复")
        ids.add(case["id"])
        cases.append(case)
        pairs[case["pair_id"]].append(case)
    if not cases:
        raise ValueError("案例集不能为空")
    for pair_id, pair in pairs.items():
        if (len(pair) != 2 or len({c["expected"] for c in pair}) != 1
                or {c["variant"] for c in pair} != {"original", "paraphrase"}):
            raise ValueError(f"语义对 {pair_id} 需要标签相同的 original/paraphrase 各一条")
    return cases


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _consistency(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    comparable = consistent = 0
    for pair in pairs.values():
        if len(pair) == 2 and all(row["baseline"] for row in pair):
            comparable += 1
            consistent += pair[0]["baseline"]["choice"] == pair[1]["baseline"]["choice"]
    return {"consistent": consistent, "comparable": comparable,
            "total_pairs": len(pairs), "agreement": _ratio(consistent, comparable)}


def run_evaluation(cases, engine):
    if not cases:
        raise ValueError("案例集不能为空")
    # Model download/load and one warmup call are outside all measured samples.
    engine.decide(cases[0]["context"])
    rows = []
    for case in cases:
        row = dict(case, baseline=None, reordered=None, error=None, reordered_error=None)
        for target, error_key, kwargs in (
            ("baseline", "error", {}),
            ("reordered", "reordered_error", {"option_order": REVERSED_ORDER}),
        ):
            try:
                row[target] = engine.decide(case["context"], **kwargs).to_dict()
            except Exception as exc:
                row[error_key] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    completed = [row for row in rows if row["baseline"]]
    matched = sum(row["baseline"]["choice"] == row["expected"] for row in completed)
    comparable = [row for row in rows if row["baseline"] and row["reordered"]]
    order_matches = sum(row["baseline"]["choice"] == row["reordered"]["choice"] for row in comparable)
    latency = [row["baseline"]["elapsed_ms"] for row in completed]
    total = len(rows)
    errors = total - len(completed)
    reordered_errors = sum(row["reordered"] is None for row in rows)
    per_label = {}
    for label in LABELS:
        subset = [row for row in rows if row["expected"] == label]
        label_matches = sum(bool(row["baseline"]) and row["baseline"]["choice"] == label for row in subset)
        per_label[label] = {"total": len(subset), "matched": label_matches,
                            "agreement": _ratio(label_matches, len(subset))}
    return {
        "status": "complete" if not errors and not reordered_errors else "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"warmup_calls": 1, "baseline_order": ["buy", "wait", "skip"],
                     "reordered": list(REVERSED_ORDER), "calibrated": False,
                     "notes": "人工虚构小样本；每两个改写是一组场景。未经校准，不代表购买正确概率。"},
        "summary": {
            "total": total, "completed": len(completed), "errors": errors,
            "matched": matched, "label_agreement": _ratio(matched, total),
            "per_label": per_label, "paraphrase": _consistency(rows),
            "option_order": {"consistent": order_matches, "comparable": len(comparable),
                             "total": total, "errors": reordered_errors,
                             "agreement": _ratio(order_matches, len(comparable))},
            "latency_ms": {"median": statistics.median(latency) if latency else None,
                           "mean": statistics.mean(latency) if latency else None,
                           "min": min(latency) if latency else None,
                           "max": max(latency) if latency else None},
        },
        "rows": rows,
    }


def _percent(value):
    return f"{value:.1%}" if value is not None else "无可用数据"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Strict JSON prevents NaN/Infinity from producing misleading report files.
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = directory / "results.json.tmp"
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(directory / "results.json")
    lines = ["# Laya 购物决策评测", "", f"- 运行状态：`{report['status']}`",
             f"- 运行时间（UTC）：{report['created_at']}",
             "- 候选概率未经校准；一致率仅表示符合本项目预设标准，不是客观购买正确率。", ""]
    if "summary" not in report:
        lines += ["本次未完成真实模型评测，没有可报告的准确率或推理耗时。", "", report.get("error", "运行失败"), ""]
    else:
        s = report["summary"]
        lines += [f"- 基准标签一致率：**{_percent(s['label_agreement'])}**（{s['matched']}/{s['total']}）。",
                  f"- 基准请求完成 {s['completed']}/{s['total']}，错误 {s['errors']}；错误计入总样本分母。",
                  f"- 等价改写一致率：{_percent(s['paraphrase']['agreement'])}（{s['paraphrase']['consistent']}/{s['paraphrase']['comparable']} 可比较组，共 {s['paraphrase']['total_pairs']} 组）。",
                  f"- 选项逆序一致率：{_percent(s['option_order']['agreement'])}（{s['option_order']['consistent']}/{s['option_order']['comparable']} 可比较条，逆序错误 {s['option_order']['errors']}）。",
                  "- 上述一致性不表示正确性：两次都答错也可能一致。"]
        if s["latency_ms"]["median"] is not None:
            lines += [f"- 预热后基准耗时：中位数 {s['latency_ms']['median']:.1f} ms，平均 {s['latency_ms']['mean']:.1f} ms；不含模型加载，含分词和前向推理。"]
        lines += ["", "## 各类结果", "", "| 预期类别 | 标签一致 | 一致率 |", "| --- | --- | --- |"]
        for label, item in s["per_label"].items():
            lines.append(f"| {LABELS[label]} | {item['matched']}/{item['total']} | {_percent(item['agreement'])} |")
        lines += ["", "## 逐例结果（包括全部错误案例）", "", "| ID | 预期 | 基准输出 | 逆序输出 | 基准符合 |", "| --- | --- | --- | --- | --- |"]
        for row in report["rows"]:
            base = LABELS[row["baseline"]["choice"]] if row["baseline"] else _cell(row["error"])
            reverse = LABELS[row["reordered"]["choice"]] if row["reordered"] else _cell(row["reordered_error"])
            match = "是" if row["baseline"] and row["baseline"]["choice"] == row["expected"] else "否"
            lines.append(f"| {_cell(row['id'])} | {LABELS[row['expected']]} | {base} | {reverse} | {match} |")
        lines += ["", "每条输入、选项说明、未经校准的分布、耗时及原始模型返回均保存在 [results.json](results.json)。", ""]
    if report.get("runtime"):
        lines += ["## 运行环境", "", "```json", json.dumps(report["runtime"], ensure_ascii=False, indent=2), "```", ""]
    temporary = directory / "README.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(directory / "README.md")


def runtime_metadata():
    from chong.runtime import configure_runtime
    configure_runtime()
    from huggingface_hub.constants import HF_HUB_CACHE
    packages = {}
    for name in ("laya", "torch", "transformers", "streamlit"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    revision_file = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya-multilingual/refs/main"
    revision = revision_file.read_text().strip() if revision_file.exists() else None
    return {"python": platform.python_version(), "os": platform.system(),
            "architecture": platform.machine(), "device": "cpu", "packages": packages,
            "checkpoint": "convaiinnovations/laya-multilingual", "checkpoint_revision": revision}


def main(argv=None):
    from chong.runtime import ROOT, configure_runtime
    configure_runtime()
    from chong.decision import DecisionEngine

    parser = argparse.ArgumentParser(description="运行真实 Laya 模型的中文购物决策评测")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latest")
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.cases)
        print(f"加载 Laya CPU 模型；预热后运行 {len(cases)} 条基准及 {len(cases)} 条逆序请求。", flush=True)
        report = run_evaluation(cases, DecisionEngine())
    except Exception as exc:
        report = {"status": "failed", "created_at": datetime.now(timezone.utc).isoformat(),
                  "error": f"{type(exc).__name__}: {exc}", "rows": []}
    report["runtime"] = runtime_metadata()
    write_report(report, args.output)
    if report["status"] == "failed":
        print(report["error"])
        return 1
    summary = report["summary"]
    print(f"标签一致：{summary['matched']}/{summary['total']}；报告：{args.output}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
