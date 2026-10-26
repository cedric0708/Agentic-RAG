"""터미널용 대화형 채팅 (웹 UI 권장: chat_web.py).

실행:
    python chat_main.py
"""

from __future__ import annotations

from agents.chat_service import create_chat, is_reset_command, send_message
from agents.config import load_config


def main() -> None:
    config = load_config()
    bundle, welcome = create_chat(config)

    print("=" * 70)
    print("💬 Agentic-RAG 대화형 복약 상담 (터미널)")
    print("ChatGPT형 웹 UI: python chat_web.py")
    print(f"ChromaDB: {'ON' if config.use_chroma else 'OFF'}")
    print(f"Reasoning trace: {'ON' if config.show_reasoning else 'OFF'} (SHOW_REASONING=false 로 끄기)")
    from agents.llm_client import is_llm_enabled
    from agents.conversation_agent import _use_llm_orchestrator
    llm_on = is_llm_enabled()
    print(f"LLM: {'ON' if llm_on else 'OFF (OPENAI_API_KEY 필요)'}")
    print(f"LLM orchestrator: {'ON' if _use_llm_orchestrator() else 'OFF'}")
    print("=" * 70)
    print()
    print(f"Assistant: {welcome}")
    print()

    while bundle.session.phase.value != "ended":
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: 상담을 종료할게요. 증상이 지속되면 병원·약국 상담을 권해드려요.")
            break

        if not user_text:
            continue

        if is_reset_command(user_text):
            bundle, welcome = create_chat(config)
            print(f"\nAssistant: {welcome}\n")
            continue

        bundle, reply, trace, _sources = send_message(bundle, user_text)

        if trace and config.enable_agent_trace:
            print(f"\n--- [Debug] ---\n{trace}\n")

        print(f"Assistant: {reply}\n")

    print("=" * 70)


if __name__ == "__main__":
    main()
