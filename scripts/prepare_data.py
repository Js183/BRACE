"""Repack CMUMOSI_new.pkl / CMUMOSEI_new.pkl into the split arrays read by data.py,
without changing features, labels or official splits.

    python scripts/prepare_data.py --input CMUMOSEI_new.pkl --output data/mosei.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

FIELDS = {"text": "text_features", "audio": "audio_features", "vision": "visual_features"}
SPLITS = (("train", "train"), ("val", "valid"), ("test", "test"))


def prepare(source, destination):
    with Path(source).open("rb") as f:
        raw = pickle.load(f, encoding="latin1")
    output, counts = {}, {}
    for source_split, split in SPLITS:
        ids, labels, texts = [], [], []
        features = {name: [] for name in FIELDS}
        for video in sorted(raw[source_split]):
            for index, segment in enumerate(raw["video_ids"][video]):
                ids.append(str(segment))
                labels.append(raw["labels"][video][index])
                texts.append(str(raw["texts"][video][index]))
                for name, field in FIELDS.items():
                    features[name].append(np.asarray(raw[field][segment], dtype=np.float32))
        count = len(ids)
        arrays = {
            "id": np.asarray(ids),
            "raw_text": np.asarray(texts),
            "regression_labels": np.asarray(labels, dtype=np.float32).reshape(-1, 1),
        }
        for name, values in features.items():
            arrays[name] = np.stack(values, axis=0)[:, None, :]
            arrays[name + "_lengths"] = np.ones(count, dtype=np.int64)
        output[split] = arrays
        counts[split] = count
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        pickle.dump(output, f, protocol=4)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output)))
