"""Token counting helpers for chat messages."""
import tiktoken

ENCODING_NAME = "cl100k_base"
enc = tiktoken.get_encoding(ENCODING_NAME)

def ntok(text: str) -> int:
    return len(enc.encode(text or ""))

def message_tokens(message: dict) -> int:
    # Approximate chat-format overhead; exact API accounting can differ.
    return ntok(message.get("content", "")) + 4

def count_messages(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages) + (3 if messages else 0)

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a security assistant."},
        {"role": "user", "content": "Summarise this fictional pentest log."},
        {"role": "assistant", "content": "Here is the summary."},
    ]
    for item in messages:
        print(f"{item['role']:9} {ntok(item['content']):4} content tokens")
    print("Approx total:", count_messages(messages), "tokens")
