import shutil
from pathlib import Path


def main():
    src = Path("out/model.pkl")
    cur = Path("deploy/current")
    bak = Path("deploy/last")
    cur.mkdir(parents=True, exist_ok=True)
    bak.mkdir(parents=True, exist_ok=True)

    if (cur / "model.pkl").exists():
        shutil.copy2(cur / "model.pkl", bak / "model.pkl")

    shutil.copy2(src, cur / "model.pkl")
    print("done")


if __name__ == "__main__":
    main()
