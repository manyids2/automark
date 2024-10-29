from pathlib import Path
import pickle
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="path to pickle")
    args = parser.parse_args()

    with open(args.path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except TypeError:
            data = pickle.load(f)

    keys = list(data.keys())
    for k in keys:
        subkeys = list(data[k].keys())
        # # assert set(subkeys) == set(["inputs", "outputs"])
        assert len(data[k]["inputs"]) == len(data[k]["outputs"])
        print(f"{k:21s} : {list(data[k]["inputs"][0].keys())} ({len(data[k]["inputs"])})")
