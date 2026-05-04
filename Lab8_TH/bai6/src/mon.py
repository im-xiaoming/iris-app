import json
from pathlib import Path


def main():
    p = Path("monitor/pred.jsonl")
    if not p.exists():
        print("n 0")
        return
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = len(rows)
    ms = round(sum(x["ms"] for x in rows) / n, 3) if n else 0
    print(f"n {n}")
    print(f"ms {ms}")


if __name__ == "__main__":
    main()
