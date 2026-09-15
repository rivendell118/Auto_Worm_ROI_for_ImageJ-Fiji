from __future__ import print_function

import argparse
import os
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Finalize the validated 0.3.3 low-clarity checkpoint")
    parser.add_argument("candidate")
    parser.add_argument("output")
    args = parser.parse_args()

    checkpoint = torch.load(args.candidate, map_location="cpu")
    checkpoint.update({
        "postprocess_interior_threshold": None,
        "postprocess_erosion": 3,
        "postprocess_min_area": 0.003,
        "postprocess_min_height": 0.10,
        "postprocess_max_instances": 12,
        "normalization_mode": "background_aware",
        "software_version": "0.3.3",
        "training_purpose": "low_clarity_adjacent_boundary_recognition",
        "training_dataset": "base_v020_76_plus_supplement3_20",
        "supplement_validation_count_accuracy": "8/10",
        "supplement_validation_mean_matched_dice": 0.8949871040040082,
        "old76_regression_count_accuracy": "72/76",
    })
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="worm_033_", suffix=".pt", dir=os.path.dirname(output))
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        # Reload before replacement so a serialization failure cannot damage the
        # model currently used by the 0.3.3 working copy.
        verified = torch.load(temporary, map_location="cpu")
        if "model_state" not in verified or verified.get("software_version") != "0.3.3":
            raise ValueError("Final checkpoint verification failed")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Final model:", output)


if __name__ == "__main__":
    main()
