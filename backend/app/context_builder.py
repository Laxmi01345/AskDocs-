MAX_CONTEXT_TOKENS = 6000
SUMMARY_TRIGGER_TOKENS = 3000
RECENT_WINDOW_SIZE = 5


def estimate_tokens(text):
    return len(text) // 4


def build_rag_prompt(session, question, rag_context):
    parts = []

    parts.append(
        "You are a helpful AI assistant that answers questions about documents. "
        "Use the provided context to answer questions accurately. "
        "Consider the conversation history when answering follow-up questions. "
        "If the answer is not in the context, say you don't know."
    )
    parts.append(f"\nDocument Context:\n{rag_context}")

    if session.turns:
        history = _get_bounded_history(session)
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"\n{content}")
            elif role == "human":
                parts.append(f"\nHuman: {content}")
            elif role == "ai":
                parts.append(f"\nAssistant: {content}")

    parts.append(f"\nHuman: {question}")
    parts.append("\nAssistant:")

    return "\n".join(parts)


def _get_bounded_history(session):
    messages = []
    current_tokens = 0

    if session.summary:
        messages.append({"role": "system", "content": f"[Conversation summary]\n{session.summary}"})
        current_tokens += estimate_tokens(session.summary)

    recent_turns = session.turns[-RECENT_WINDOW_SIZE:]
    for turn in reversed(recent_turns):
        turn_tokens = estimate_tokens(turn.user_message) + estimate_tokens(turn.assistant_message)
        if current_tokens + turn_tokens > MAX_CONTEXT_TOKENS - 1000:
            break
        messages.insert(0, {"role": "ai", "content": turn.assistant_message})
        messages.insert(0, {"role": "human", "content": turn.user_message})
        current_tokens += turn_tokens

    return messages


def should_summarize(session):
    if session.turn_count <= RECENT_WINDOW_SIZE:
        return False
    total_tokens = sum(
        estimate_tokens(t.user_message) + estimate_tokens(t.assistant_message)
        for t in session.turns
    )
    return total_tokens > SUMMARY_TRIGGER_TOKENS


def build_summarization_prompt(session):
    turns_to_summarize = session.turns[:-RECENT_WINDOW_SIZE]
    lines = []
    for turn in turns_to_summarize:
        lines.append(f"Human: {turn.user_message}")
        lines.append(f"Assistant: {turn.assistant_message}")
    transcript = "\n".join(lines)

    return (
        "Summarize the following conversation in 5-8 sentences, "
        "preserving proper nouns, numbers, user preferences, "
        "unresolved questions, and technical details. "
        "Drop greetings and small talk.\n\n"
        f"---\n{transcript}\n---\n\nSummary:"
    )
