def is_hallucination(response: str, expected_keywords=None) -> bool:
    if expected_keywords is None:
        return False
    response_lower = response.lower()
    return not any(k.lower() in response_lower for k in expected_keywords)
