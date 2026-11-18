import os
from pathlib import Path
from typing import cast

import streamlit as st
# from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputParam

# ENV_PATH = Path(r"C:\env\.env")
DEFAULT_MODEL = "gpt-4o-mini"
MODELS = ["gpt-4o-mini", "gpt-4o"]
DEFAULT_SYSTEM_PROMPT = (
    "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. "
    "사용자의 질문에 정확하고 유용한 답변을 제공하세요."
)

# load_dotenv(ENV_PATH)

st.set_page_config(
    page_title="AI 챗봇",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("messages", [])


@st.cache_resource
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def ask_model(
    messages: list[EasyInputMessageParam],
    *,
    model: str,
    temperature: float,
    instructions: str,
) -> str:
    response = get_client().responses.create(
        model=model,
        input=cast(ResponseInputParam, messages),
        instructions=instructions,
        temperature=temperature,
    )
    return response.output_text


def message_text(message: EasyInputMessageParam) -> str:
    content = message["content"]
    return content if isinstance(content, str) else ""


def show_message(role: str, content: str, key: str) -> None:
    alignment = "right" if role == "user" else "left"
    with st.container(horizontal=True, horizontal_alignment=alignment):
        with st.container(key=key):
            st.markdown(content)


with st.sidebar:
    st.markdown("### :material/settings: 설정")
    model = st.selectbox("모델 선택", MODELS, index=MODELS.index(DEFAULT_MODEL))
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        format="%.2f",
        help="값이 높을수록 답변이 더 다양해집니다.",
    )
    if st.button(
        "대화 초기화",
        icon=":material/mop:",
        width="stretch",
        key="reset_chat",
    ):
        st.session_state.messages = []
        st.rerun()

    st.space("small")
    st.markdown("#### :material/description: 시스템 프롬프트")
    st.caption("시스템 프롬프트를 수정할 수 있습니다.")
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value=DEFAULT_SYSTEM_PROMPT,
        height=150,
        label_visibility="collapsed",
        key="system_prompt",
    )

st.html(
    """
    <style>
    [class*="st-key-user-"],
    [class*="st-key-assistant-"] {
        width: max-content;
        max-width: min(34rem, 78%);
        border-radius: 16px;
        padding: 0.15rem 0.9rem;
    }
    [class*="st-key-user-"] {
        margin-left: auto;
        background: #f26b5a;
    }
    [class*="st-key-assistant-"] {
        margin-right: auto;
        background: #f3f4f6;
    }
    [class*="st-key-user-"] p,
    [class*="st-key-assistant-"] p {
        margin: 0.45rem 0;
    }
    [class*="st-key-user-"] p {
        color: #ffffff;
    }
    .chat-heading {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.2rem 0 0.15rem;
    }
    .chat-logo {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: #6d5efc;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        line-height: 1;
        flex: none;
    }
    .chat-title {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        color: #1f2937;
    }
    </style>
    <div class="chat-heading">
        <div class="chat-logo">🤖</div>
        <div class="chat-title">AI 챗봇</div>
    </div>
    """
)
st.caption("OpenAI API를 사용한 대화형 AI 어시스턴트")

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        f"`{ENV_PATH}` 파일에서 OPENAI_API_KEY를 읽지 못했습니다.",
        title="API 키 없음",
    )
    st.stop()

history: list[EasyInputMessageParam] = st.session_state.messages
for index, message in enumerate(history):
    show_message(message["role"], message_text(message), f"{message['role']}-{index}")

prompt = st.chat_input("메시지를 입력하세요...", submit_mode="disable")
if prompt:
    history.append({"role": "user", "content": prompt})
    show_message("user", prompt, f"user-{len(history) - 1}")
    try:
        with st.spinner("응답을 생성하는 중"):
            answer = ask_model(
                history,
                model=model,
                temperature=temperature,
                instructions=system_prompt,
            )
    except Exception:
        st.error("모델 응답을 가져오지 못했습니다.", title="요청 실패")
    else:
        history.append({"role": "assistant", "content": answer})
        show_message("assistant", answer, f"assistant-{len(history) - 1}")
