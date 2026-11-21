"""Keep the system message and the newest conversation messages."""
from account import message_tokens

def fit_window(messages: list[dict], budget: int) -> tuple[list[dict], int]:
    if not messages:
        return [], 0

    system = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if system else messages[:]

    used = message_tokens(system) if system else 0
    kept = []

    # Add newest messages first, then restore chronological order.
    for message in reversed(rest):
        cost = message_tokens(message)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost

    result = ([system] if system else []) + list(reversed(kept))
    return result, used
