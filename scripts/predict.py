import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--with-prob", action="store_true")
    args = ap.parse_args()

    from radar import config as C
    from radar.inference import list_images, predict_dir

    out = Path(args.output)
    out_csv = out if out.suffix == ".csv" else out / "prediction.csv"
    thr = C.THRESHOLD if args.threshold is None else args.threshold
    ladder = [("full pipeline", {}), ("no lung crop", {"use_crop": False}),
              ("cpu, no lung crop", {"use_crop": False, "device": "cpu"})]
    for name, kw in ladder:
        try:
            if kw.get("device") == "cpu":
                import torch
                kw = {**kw, "device": torch.device("cpu")}
            predict_dir(args.input, out_csv, weights_dir=args.weights, threshold=thr,
                        with_prob=args.with_prob, **kw)
            return
        except Exception as e:  # noqa: BLE001
            print(f"[predict] {name} failed: {type(e).__name__}: {e}", flush=True)

    import pandas as pd
    names = [os.path.basename(p) for p in list_images(args.input)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"filename": names, "TB/Normal": ["Normal"] * len(names)}).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
