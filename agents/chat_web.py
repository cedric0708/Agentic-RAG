"""ChatGPT형 웹 채팅 UI (Gradio).

실행:
    pip install gradio
    python chat_web.py

브라우저: http://127.0.0.1:7860
"""

from __future__ import annotations

import gradio as gr

from agents.chat_service import ChatBundle, create_chat, is_reset_command, send_message
from agents.config import load_config


def _history_to_messages(history: list) -> list[dict]:
    """Gradio messages 형식으로 정규화."""
    if not history:
        return []
    if isinstance(history[0], dict):
        return history
    # legacy tuples → messages
    messages = []
    for user, assistant in history:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    return messages


def respond(user_message: str, history: list, bundle: ChatBundle | None):
    history = _history_to_messages(history)

    if bundle is None:
        bundle, welcome = create_chat()
        history = [{"role": "assistant", "content": welcome}]

    text = (user_message or "").strip()
    if not text:
        return history, bundle, ""

    if is_reset_command(text):
        bundle, welcome = create_chat()
        return [{"role": "assistant", "content": welcome}], bundle, ""

    bundle, reply, _trace, _sources, _reasoning = send_message(bundle, text)
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply},
    ]
    return history, bundle, ""


def new_chat():
    bundle, welcome = create_chat()
    return [{"role": "assistant", "content": welcome}], bundle, ""


def build_ui() -> gr.Blocks:
    config = load_config()
    title = "💊 Agentic-RAG 복약 상담"
    subtitle = (
        "증상을 편하게 말씀해 주세요. 대화를 통해 정보를 확인한 뒤 "
        "일반의약품 후보를 추천해 드립니다. (의료 진단 대체 불가)"
    )

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="teal"),
        css="""
        .chat-wrap { max-width: 900px; margin: 0 auto; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(f"# {title}\n{subtitle}")

        with gr.Column(elem_classes="chat-wrap"):
            chatbot = gr.Chatbot(
                label="상담",
                height=520,
                type="messages",
                show_copy_button=True,
                avatar_images=(None, None),
            )
            bundle_state = gr.State(None)

            with gr.Row():
                msg = gr.Textbox(
                    label="메시지",
                    placeholder="예: 머리가 아파요 / 추천해 주세요 / 왜 이 약인가요?",
                    scale=5,
                    lines=1,
                    max_lines=4,
                )
                send_btn = gr.Button("전송", variant="primary", scale=1)
                reset_btn = gr.Button("새 대화", scale=1)

            gr.Markdown(
                f"**ChromaDB:** {'ON' if config.use_chroma else 'OFF'} · "
                "**팁:** 자연스럽게 증상을 말하고, 이어서 질문에 답해 주세요."
            )

        # 첫 환영 메시지
        def _boot():
            b, welcome = create_chat()
            return [{"role": "assistant", "content": welcome}], b

        demo.load(_boot, outputs=[chatbot, bundle_state])

        submit_args = dict(
            fn=respond,
            inputs=[msg, chatbot, bundle_state],
            outputs=[chatbot, bundle_state, msg],
        )
        send_btn.click(**submit_args)
        msg.submit(**submit_args)
        reset_btn.click(fn=new_chat, outputs=[chatbot, bundle_state, msg])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
