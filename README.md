# 今天你冲了吗？

### A local lab for exploring Laya's shopping decisions

**Describe a purchase, then inspect Laya's buy / wait / skip choice, uncalibrated probabilities, inference time, and raw request and response.**

**输入用途、预算和替代品等购物情况，在本地观察 Laya 如何选择「冲 / 等等 / 不冲」，并查看未经校准的候选概率、推理耗时及原始请求与返回。**

[English](README.en.md) | 简体中文

[快速开始](#快速开始) · [基准结果](#基准结果) · [完整评测报告](reports/baseline/README.md)

这是一个可检查模型行为的学习项目，不提供可靠的购物建议。页面中的概率只表示模型在三个选项间的相对倾向，不代表建议正确的概率。

![真实基准记录 buy-07 的输入、模型选择和候选概率可视化](docs/images/demo.png)

图为真实推理记录 `buy-07` 的可视化示例，**不是应用界面截图**。原始请求与返回可在[机器可读报告](reports/baseline/results.json)中核对。

## 快速开始

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。以下命令适用于 Windows PowerShell 和 CPU；首次运行会从 Hugging Face 下载公开模型权重，约需 650 MB 缓存空间，无需 API Key。

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

在浏览器打开终端显示的本地地址（通常是 <http://localhost:8501>），选一个预设情境或输入自己的购物情况，再点击「冲不冲」。模型在当前页面会话中复用；修改输入或切换预设情境会清除旧结果。首次加载模型通常比后续推理慢。

`requirements.txt` 固定应用的 `laya==0.3.20` 和 `streamlit==1.64.0`；`requirements-lock.txt` 是生成基准报告时 Windows CPU 环境的完整依赖快照，不能直接视为跨平台锁文件。macOS/Linux 请将虚拟环境中的可执行文件路径换成 `.venv/bin/python`，并按自己的平台安装 PyTorch。

权重下载完成后，可以设置 `HF_HUB_OFFLINE=1` 再启动应用以离线运行；尚未缓存权重时不要启用离线模式。默认 Hugging Face 缓存位于项目内的 `.cache/huggingface`，除非已设置 `HF_HOME`。

## 模型做了什么

项目通过 [Laya 官方 SDK](https://github.com/NandhaKishorM/laya) 的 `choice` 接口调用 [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)；不调用外部推理 API。模型卡标注约 322M 参数。输入的购物情况和以下标准一同送入模型，由模型返回选择与候选概率；应用没有用固定规则覆盖它的选择。

| 返回值 | 页面文字 | 提供给模型的判断标准 |
| --- | --- | --- |
| `buy` | 冲 | 用途明确、预算充足，现有物品不能满足需要 |
| `skip` | 不冲 | 存在预算压力；或替代品足够且主要受促销驱动 |
| `wait` | 等等 | 其余情况，包括关键条件不清楚 |

标准要求优先考虑预算压力；「有替代品」本身不直接等于「不冲」。这些只是模型收到的选项说明，不保证模型遵循。Laya 的[模型限制说明](https://huggingface.co/convaiinnovations/laya-multilingual#limits)指出多语言权重未经校准，可能过度自信。本项目未微调、未校准，也不生成决策理由。

页面支持最多 **500 个字符**；推理前还会使用实际 tokenizer 检查 token 长度，避免输入被截断。空白、过长或超出 token 预算的输入会报 `InputError`；模型加载失败报 `ModelLoadError`；推理失败或响应无效报 `InferenceError`，不会生成替代决定。

也可从 Python 调用：

```python
from chong.decision import decide

result = decide("旧键盘多个按键失灵，每天写文档需要用；没有备用，预算已经留好。")
print(result.choice)          # 模型实际选择：buy / wait / skip
print(result.probabilities)   # 三个未经校准的候选概率
print(result.elapsed_ms)      # 分词与推理耗时，不含模型加载
print(result.to_dict())       # 还包括 model、request、raw_response
```

`decide(context, *, engine=None, option_order=None)` 可以传入可复用的 `DecisionEngine`；`option_order` 可指定三个选项的排列，供顺序敏感性评测使用。

## 基准结果

2026-09-26 在 Windows、Python 3.11.14、CPU 上，模型预热一次后运行 30 条基准请求及其 30 条选项逆序请求。30 条文本来自 **15 组虚构购物场景的等价改写**，预期标签在运行前按[人工标注规则](data/README.md)写入。

![基准结果图：预期标签一致率与改写、逆序稳定性](docs/images/benchmark.png)

| 指标 | 结果 |
| --- | ---: |
| 与预期标签一致 | **11/30（36.7%）** |
| 预期「冲 / 等等 / 不冲」分别命中 | **7/10 · 4/10 · 0/10** |
| 等价改写选择一致 | 9/15 组（60.0%） |
| 选项逆序选择一致 | 27/30 条（90.0%） |
| 预热后基准推理耗时中位数 | 145.9 ms |
| 基准和逆序推理错误 | 0 |

当前模型在这组中文案例上尤其未命中预期「不冲」的情境，含预算压力的输入也可能被选为「冲」。改写和逆序指标衡量的是选择是否一致，**不衡量选择是否正确**。这是一组小型自建案例，不代表普遍的购物决策能力；耗时也会随硬件、运行环境和模型版本改变。

阅读[完整报告与逐例结果](reports/baseline/README.md)、[机器可读报告（含请求和原始响应）](reports/baseline/results.json)、[30 条案例](data/cases.jsonl)。本次记录的模型 revision 为 `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`。

重新评测会默认写入 `reports/latest`，不覆盖随项目提供的基准报告：

```powershell
.venv/Scripts/python.exe -m chong.evaluation
# 或指定输出目录
.venv/Scripts/python.exe -m chong.evaluation --output reports/local-my-run
```

报告会保留请求失败记录；标签一致率以全部基准请求为分母，改写和逆序一致率只统计可比较的项目。若模型加载失败，评测标记为 `failed`，不会虚构准确率。

## 测试与项目结构

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
```

自动测试用 SDK 替身检查接口、统计和页面交互，不下载模型；随附基准报告则由真实权重生成。

安装开发依赖后，可从随附基准记录重新生成上面的两张图片：

```powershell
.venv/Scripts/python.exe scripts/render_examples.py
```

```text
app.py                  Streamlit 页面
chong/decision.py        Laya 推理封装与输入校验
chong/evaluation.py      案例评测和报告导出
chong/runtime.py         本地缓存配置
data/                   案例与标注说明
reports/baseline/        基准报告和逐例响应
docs/images/            推理示例与评测图片
scripts/                从基准报告生成示例图片
tests/                  自动测试
```

应用只在当前会话保留输入和结果，不保存购物历史；首次下载权重需要访问 Hugging Face。页面默认供本地使用，避免把输入中的私人信息放入公开截图或评测案例。[Laya 代码](https://github.com/NandhaKishorM/laya)及[模型](https://huggingface.co/convaiinnovations/laya-multilingual)采用 Apache-2.0 许可；本仓库不包含模型权重。[jevlike](https://github.com/vinnylarouge/jevlike) 的「上下文 + 候选选项」实验启发了这里的接口设计。

维护者：[@cloudwallker](https://github.com/cloudwallker)
