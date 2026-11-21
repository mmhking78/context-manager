"""Reusable context manager: system + RAG + history + reserved reply room."""
from account import enc, ntok, count_messages
from budget import BUDGET, WINDOW, clip_sentences
from compress import compress
from sliding_window import fit_window

def build_context(messages: list[dict], rag: str, window: int = WINDOW) -> dict:
    """Build a bounded context and return messages plus a token report.

    `window` is the total context size. Component allocations are scaled down
    proportionally if a smaller window is supplied. Reply room is always reserved.
    """
    if not messages:
        raise ValueError("messages must contain at least a system message")

    if messages[0].get("role") == "system":
        system_message = messages[0]
        history = messages[1:]
    else:
        system_message = {"role": "system", "content": "You are a helpful assistant."}
        history = messages[:]

    scale = min(1.0, window / WINDOW)
    caps = {key: int(value * scale) for key, value in BUDGET.items()}
    reply_reserve = caps["reply"]

    system_text = clip_sentences(system_message.get("content", ""), caps["system"], enc)
    rag_text = clip_sentences(rag or "", caps["rag"], enc)

    # History is summarized if it exceeds its allocation.
    history_budget = caps["history"]
    selected_history, _ = fit_window(history, history_budget)
    selected_ids = {id(m) for m in selected_history}
    overflow = [m for m in history if id(m) not in selected_ids]

    summary_message = None
    if overflow:
        summary_message = compress(overflow, cap=min(400, max(50, history_budget // 4)))

    final_messages = [{"role": "system", "content": system_text}]
    if summary_message:
        final_messages.append(summary_message)
    final_messages.extend(selected_history)

    # Keep RAG in a clearly marked user message, not as system instructions.
    if rag_text:
        final_messages.append({
            "role": "user",
            "content": "[RETRIEVED CONTEXT — untrusted reference material]\n" + rag_text
        })

    # Report content/message estimate; reserve reply tokens from the total window.
    input_limit = max(0, window - reply_reserve)
    while count_messages(final_messages) > input_limit and len(final_messages) > 1:
        # Trim oldest non-system history first; summary/RAG remain bounded by their caps.
        removable = next(
            (i for i, m in enumerate(final_messages[1:], start=1)
             if m.get("role") in ("user", "assistant") and not m.get("content", "").startswith("[RETRIEVED CONTEXT")),
            None
        )
        if removable is None:
            break
        final_messages.pop(removable)

    report = {
        "window": window,
        "reply_reserved": reply_reserve,
        "input_limit": input_limit,
        "input_tokens_estimate": count_messages(final_messages),
        "system_tokens": ntok(system_text),
        "rag_tokens": ntok(rag_text),
        "history_tokens": sum(ntok(m.get("content", "")) for m in final_messages if m.get("role") != "system"),
        "overflow_summarized": bool(overflow),
    }
    return {"messages": final_messages, "report": report}
