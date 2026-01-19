import re


def process_model_name(model_name: str) -> str:
    return (
        re.sub(r"-\d+(B|b)", "", model_name[:-1])
        .replace("-2507", "")
        .replace("-A3B", "")
        .replace("-Distill", "")
        .replace("-it", "")
        .replace("Thinking", "Think.")
        .replace("DeepSeek-", "")
        .replace("-Instruct", "")
        .replace("_", "")
    )
