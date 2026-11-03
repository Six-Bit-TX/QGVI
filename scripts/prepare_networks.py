#!/usr/bin/env python3
"""Load the exact upstream models once, allowing their official weight downloads.

LoGS includes SuperGlue/SuperPoint with separate noncommercial terms. This
release does not mirror those weights or relicense the dependency code.
No device is selected here: use your PyTorch/environment configuration.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/LoGS"))

if __name__ == "__main__":
    import torch
    from hloc.extractors.netvlad import NetVLAD
    from hloc.extractors.superpoint import SuperPoint
    from hloc.matchers.superglue import SuperGlue
    torch.hub.set_dir(str(ROOT / ".cache/torch"))
    for cls, config in ((NetVLAD, {}), (SuperPoint, {"nms_radius": 3, "max_keypoints": 4096}),
                        (SuperGlue, {"weights": "outdoor", "sinkhorn_iterations": 5})):
        model = cls(config)
        print(f"Prepared {cls.__name__}")
        del model
