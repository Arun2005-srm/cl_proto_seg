"""Build the image/label list for CamVid (SegNet-Tutorial layout) used by configs/camvid.yaml.

    python tools/make_camvid_lists.py --root data/camvid

CamVid has 701 frames from 4 drive sequences (0001TP, 0006R0, 0016E5, Seq05VD), which
are the domains. The official train/val/test folders split the sequences unevenly, so
all frames go into one list ("all.txt") and configs/camvid.yaml splits every sequence
70/10/20 into train/val/test.
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/camvid")
    args = ap.parse_args()
    lines = []
    for split in ("train", "val", "test"):
        for f in sorted(os.listdir(os.path.join(args.root, split))):
            if f.endswith(".png") and os.path.isfile(os.path.join(args.root, split + "annot", f)):
                lines.append(f"{split}/{f} {split}annot/{f}")
    with open(os.path.join(args.root, "all.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} pairs to {os.path.join(args.root, 'all.txt')}")


if __name__ == "__main__":
    main()
