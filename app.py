"""今天你冲了吗 — a local, inspectable Laya choice experiment."""
import streamlit as st

from chong.runtime import configure_runtime

configure_runtime()

from chong.decision import (  # noqa: E402
    DecisionEngine, InferenceError, InputError, LABELS, MAX_CONTEXT_CHARS,
    MODEL_ID, ModelLoadError,
)

st.set_page_config(page_title="今天你冲了吗 · Laya 决策实验", page_icon="🟠", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1120px; padding-top: 2.6rem; padding-bottom: 2rem; }
h1 { font-size: clamp(2.4rem, 5vw, 4rem) !important; letter-spacing: -.04em; line-height: 1.15 !important; }
h2, h3 { letter-spacing: -.025em; }
.lab-mark { font: 700 11px/1.4 monospace; letter-spacing: .16em; color: #687466; margin-bottom: 22px; }
.lab-mark span { color: #E7683C; }
.intro { color: #666D65; font-size: 17px; line-height: 1.8; max-width: 580px; margin-bottom: 28px; }
.section-label { color: #798074; font: 600 11px/1.5 monospace; letter-spacing: .14em; margin-bottom: 12px; }
.status-pill { border: 1px solid #DDDED4; border-radius: 50px; padding: 8px 15px; display: inline-block; font-size: 12px; color: #5D6C59; }
.verdict { font-size: 66px; font-weight: 800; line-height: 1.2; letter-spacing: -.07em; margin: 12px 0 12px; }
.empty-result { padding: 48px 15px 40px; text-align: center; color: #81897D; }
.empty-result .possibilities { font-size: 32px; font-weight: 700; color: #A8AEA2; letter-spacing: .08em; margin-bottom: 22px; }
.empty-result p { font-size: 14px; line-height: 1.9; }
.footer-note { border-top: 1px solid #DDDCD2; margin-top: 36px; padding-top: 17px; font: 11px/1.6 monospace; letter-spacing: .08em; color: #8A8F82; }
div[data-testid="stTextArea"] textarea { line-height: 1.8; border-radius: 12px; }
div[data-testid="stButton"] button { border-radius: 10px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
@media (max-width: 640px) { .block-container { padding-top: 1.4rem; } .verdict { font-size: 52px; } }
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "确实需要": "每天用电脑写文档，现有键盘多个按键失灵，家里没有备用键盘；更换费用已经列入预算。",
    "有点上头": "我想买一把新键盘。现在的键盘正常使用，新款主要是颜色好看。预算比较紧，但活动今晚结束。",
    "还没想好": "我想买一个小桌子放电脑，也知道家里没有合适桌子，但还没核对这笔钱是否在本月预算内。",
}


def clear_result():
    st.session_state["result"] = None
    st.session_state["problem"] = None


def use_preset(name):
    st.session_state["context"] = PRESETS[name]
    clear_result()


for key, default in (("context", ""), ("result", None), ("problem", None)):
    if key not in st.session_state:
        st.session_state[key] = default

top_left, top_right = st.columns([3, 1], vertical_alignment="center")
with top_left:
    st.markdown('<div class="lab-mark"><span>●</span> CHONG / DECISION LAB</div>', unsafe_allow_html=True)
with top_right:
    st.markdown('<div class="status-pill">Laya · 本地推理 · 中文</div>', unsafe_allow_html=True)

st.title("今天你冲了吗？")
st.markdown('<div class="intro">想买，但还在犹豫。<br>描述你的情况，看看 Laya 会在「冲 / 等等 / 不冲」里怎么选。</div>', unsafe_allow_html=True)

left, right = st.columns([1.08, 1], gap="large")
with left:
    st.markdown('<div class="section-label">01 / YOUR CONTEXT</div>', unsafe_allow_html=True)
    st.subheader("这次想买什么？")
    st.caption("说说用途、预算，以及手头有没有能替代的东西。")
    preset_columns = st.columns(3)
    for column, name in zip(preset_columns, PRESETS):
        column.button(name, key=f"preset_{name}", on_click=use_preset, args=(name,), width="stretch")
    context = st.text_area(
        "描述购物情况", key="context", height=185, max_chars=MAX_CONTEXT_CHARS,
        placeholder="比如：旧耳机坏了，每天开会需要用，预算已留好，也没有备用……",
        on_change=clear_result, label_visibility="collapsed",
    )
    submitted = st.button("冲不冲 →", key="decide", type="primary", width="stretch")
    st.caption("首次点击会下载并加载模型；之后复用。下载完成后可离线运行。")

    if submitted:
        clear_result()
        if not context.strip():
            st.session_state["problem"] = ("warning", "请输入购物情况，或先选一个上方示例。")
        else:
            if "engine" not in st.session_state:
                st.session_state["engine"] = DecisionEngine()
            try:
                with st.spinner("Laya 正在比较三个选项……首次加载可能需要稍等。"):
                    st.session_state["result"] = st.session_state["engine"].decide(context).to_dict()
            except InputError as exc:
                st.session_state["problem"] = ("warning", str(exc))
            except (ModelLoadError, InferenceError) as exc:
                st.session_state["problem"] = ("error", str(exc))
    if st.session_state["problem"]:
        kind, message = st.session_state["problem"]
        getattr(st, kind)(message)

with right:
    st.markdown('<div class="section-label">02 / MODEL DECISION</div>', unsafe_allow_html=True)
    with st.container(border=True):
        result = st.session_state["result"]
        if result is None:
            st.markdown('''<div class="empty-result"><div class="possibilities">冲 · 等等 · 不冲</div>
            <p>一个情境，三个选项。<br>点击「冲不冲」，在这里查看模型的选择。</p></div>''', unsafe_allow_html=True)
        else:
            choice = result["choice"]
            color = {"buy": "#42775E", "wait": "#AE7A24", "skip": "#C85C39"}[choice]
            st.caption("这一次，模型选择")
            st.markdown(f'<div class="verdict" style="color:{color}">{LABELS[choice]}</div>', unsafe_allow_html=True)
            st.caption(f"本次推理 {result['elapsed_ms']:.0f} ms · 不含模型加载")
            st.divider()
            for key, label in LABELS.items():
                probability = result["probabilities"][key]
                st.markdown(f"**{label}** {'← 模型选择' if key == choice else ''}　{probability:.1%}")
                st.progress(min(1.0, max(0.0, probability)))
        st.caption("模型概率未经校准，仅展示模型倾向，不代表“买对的概率”。")

with st.expander("查看决策标准", expanded=False):
    st.markdown("""
| 选项 | 预设标准 |
| --- | --- |
| 冲 | 用途明确、预算充足，现有物品不能满足需要 |
| 不冲 | 存在预算压力；或替代品足够，且主要受促销驱动 |
| 等等 | 其余情况，包括关键条件不清楚 |

这些标准写进模型的选项说明。预算压力优先；实际选择由 Laya 返回。
""")

if st.session_state["result"] is not None:
    with st.expander("展开本次实验：输入、选项与原始返回", expanded=False):
        result = st.session_state["result"]
        st.caption(f"模型：{result['model']} · CPU · 单次 choice")
        request_tab, response_tab = st.tabs(["实际请求", "原始返回"])
        with request_tab:
            st.json(result["request"], expanded=True)
        with response_tab:
            st.json(result["raw_response"], expanded=True)

st.markdown('<div class="footer-note">BUILT TO LEARN · LAYA MULTILINGUAL · 3 CHOICES, ONE DECISION</div>', unsafe_allow_html=True)
st.caption("学习实验 · 输入仅用于本地本次推理，应用不保存购物记录。")
