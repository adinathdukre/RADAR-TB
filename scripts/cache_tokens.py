import argparse

from radar import config as C
from radar import data as D
from radar.backbone import RadDino
from radar.external import proxy_df
from radar.segmentation import load_bboxes
from radar.tokens import cache_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=C.RES)
    ap.add_argument("--splits", nargs="+", default=["train", "test", "montgomery", "shenzhen"])
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    bb = RadDino()
    for s in args.splits:
        if s in ("train", "test"):
            df = D.load_df(C.TRAIN_CSV if s == "train" else C.TEST_CSV)
            df = df[df["label"].notna()].reset_index(drop=True)
            paths = [str(D.image_path(i, s)) for i in df["new_id"]]
        else:
            df = proxy_df(s)
            paths = df["path"].tolist()
        cache_tokens(bb, s, df["new_id"].tolist(), paths, load_bboxes(s),
                     df["label"].astype(int).tolist(), res=args.res, batch=args.batch)


if __name__ == "__main__":
    main()
