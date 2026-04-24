
def format_run_title(run: str) -> str:
    policy = neutral = negative = None
    tokens = run.split("_")
    i = 0

    while i < len(tokens):
        t = tokens[i]
        if t.startswith("policy-"):
            policy = t.replace("policy-", "")

        elif t.startswith("neutral-"):
            neutral = t.replace("neutral-", "")

        elif t == "negative" and i + 1 < len(tokens):
            if tokens[i + 1].startswith("potents-"):
                negative = tokens[i + 1].replace("potents-", "")
                i += 1
        i += 1

    if policy is None or neutral is None or negative is None:
        raise ValueError(f"The run could not be parsed correctly: {run}")

    return f"Policy: {policy} | Neutral: {neutral} | Negative: {negative}"