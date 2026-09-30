import argparse

from radar import config as C
from radar import data as D
from radar.external import proxy_df
from radar.segmentation import LungSegmenter, compute_bboxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "test", "montgomery", "shenzhen"])
    args = ap.parse_args()
    seg = LungSegmenter().load()
    for s in args.splits:
        if s in ("train", "test"):
            df = D.load_df(C.TRAIN_CSV if s == "train" else C.TEST_CSV)
            paths = [str(D.image_path(i, s)) for i in df["new_id"]]
        else:
            df = proxy_df(s)
            paths = df["path"].tolist()
        compute_bboxes(df["new_id"].tolist(), paths, s, seg)


if __name__ == "__main__":
    main()
