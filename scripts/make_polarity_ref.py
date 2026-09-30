import numpy as np
import torch

from radar import config as C
from radar import preprocessing as P
from radar.backbone import RadDino

bb = RadDino()
files = sorted(C.TRAIN_IMG_DIR.glob("*.png"))
rng = np.random.RandomState(0)
files = [files[i] for i in rng.choice(len(files), size=min(600, len(files)), replace=False)]

embs = []
with torch.no_grad():
    for k in range(0, len(files), 32):
        xs = [bb.preprocess(P.prepare_image(P.load_raw(f), square=True)) for f in files[k:k + 32]]
        _, e = bb.forward(torch.stack(xs))
        embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())

ref = np.concatenate(embs).mean(0)
ref /= np.linalg.norm(ref)
C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
np.save(C.WEIGHTS_DIR / C.POLARITY_REF, ref.astype(np.float32))
print(f"wrote {C.WEIGHTS_DIR / C.POLARITY_REF} from {len(files)} images")
