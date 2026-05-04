import json
from pathlib import Path


def main():
    p = Path("out/metrics.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    ok = d["acc"] >= 0.8
    print(f"acc {d['acc']}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
