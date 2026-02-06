def is_refusal(response: str) -> bool:
    refusal_phrases = [
        "i don't know",
        "i do not know",
        "i don't have enough information",
        "cannot answer",
        "not enough information"
    ]
    response_lower = response.lower()
    return any(p in response_lower for p in refusal_phrases)
