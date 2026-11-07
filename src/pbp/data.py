"""Load WildChat-1M into a simple table: user, timestamp, conversation turns."""


def iter_wildchat(limit=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        user_turns = []
        for turn in row["conversation"]:
            if turn.get("role") == "user":
                user_turns.append(turn.get("content") or "")
        yield {
            "user": row["hashed_ip"],
            "timestamp": row["timestamp"],
            "user_turns": user_turns,
        }
