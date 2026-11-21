"""Per-component token budgets and sentence-boundary clipping."""
from account import enc, ntok

WINDOW = 8000
BUDGET = {
    "system": 600,
    "rag": 3000,
    "history": 3000,
    "reply": 1400,  # reserve room for the model's answer
}
assert sum(BUDGET.values()) <= WINDOW

def clip_sentences(text: str, cap: int, tokenizer=enc) -> str:
    """Keep complete sentences where possible; never exceed cap tokens."""
    if cap <= 0 or not text:
        return ""
    if len(tokenizer.encode(text)) <= cap:
        return text

    # Split after common sentence-ending punctuation.
    import re
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    kept = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence]).strip()
        if len(tokenizer.encode(candidate)) <= cap:
            kept.append(sentence)
        else:
            break

    if kept:
        return " ".join(kept)

    # A single sentence is longer than the cap. Fall back to a token-safe cut.
    return tokenizer.decode(tokenizer.encode(text)[:cap])

def component_report(name: str, before: str, after: str) -> None:
    print(f"{name:10} before={ntok(before):5} after={ntok(after):5} tokens")
