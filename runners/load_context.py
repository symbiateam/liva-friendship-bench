BASE_PROMPT = "You are a friend in a conversation. ALWAYS respond in the SAME language as the user."


def get_system_prompt(sample_dir: str, task: str) -> str:
    return BASE_PROMPT
