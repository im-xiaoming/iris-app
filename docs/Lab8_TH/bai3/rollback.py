import shutil
from pathlib import Path


def main():
    cur = Path("deploy/current/model.pkl")
    bak = Path("deploy/last/model.pkl")
    if bak.exists():
        cur.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bak, cur)
        print("done")
    else:
        print("skip")


if __name__ == "__main__":
    main()
