import argparse

from radar import config as C
from radar.utils import get_logger

log = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-repo", default=None)
    ap.add_argument("--proxies", nargs="*", default=[])
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel
    AutoImageProcessor.from_pretrained(C.HF_ID)
    AutoModel.from_pretrained(C.HF_ID)
    log.info("RAD-DINO ready")

    from radar.segmentation import set_xrv_cache
    set_xrv_cache()
    import torchxrayvision as xrv
    xrv.baseline_models.chestx_det.PSPNet()
    log.info("PSPNet ready")

    if args.weights_repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.weights_repo, local_dir=str(C.WEIGHTS_DIR))
        log.info("RADAR weights -> %s", C.WEIGHTS_DIR)

    from radar.external import proxy_df
    for name in args.proxies:
        proxy_df(name)


if __name__ == "__main__":
    main()
