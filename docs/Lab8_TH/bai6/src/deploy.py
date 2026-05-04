import shutil
from pathlib import Path


def main():
    src = Path("deploy/live/model.pkl")
    dst = Path("deploy/app/model.pkl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("done")


if __name__ == "__main__":
    main()
