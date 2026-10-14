"""BiRefNet 분할 워커 (별도 파이썬에서 실행): 이미지 폴더 → 확률 마스크 PNG.

python segment_worker.py <images_dir> <out_dir> <model> <size>
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation


def main() -> int:
    src, out, model_name, size = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(dev, dtype).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    print(f"model {model_name} on {dev}: {time.time() - t0:.1f}s", flush=True)

    def load(p: Path):
        im = Image.open(p).convert("RGB")
        return p, im.size, np.asarray(im.resize((size, size), Image.BILINEAR))

    def save(p: Path, wh, prob: np.ndarray):
        Image.fromarray((prob * 255).astype(np.uint8)).resize(wh, Image.BILINEAR).save(out / f"{p.name}.png")

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    with ThreadPoolExecutor(4) as io:  # 디코딩·저장을 추론과 겹침
        for p, wh, arr in io.map(load, files):
            x = torch.from_numpy(arr.copy()).to(dev).permute(2, 0, 1)[None].float() / 255
            with torch.no_grad():
                prob = model(((x - mean) / std).to(dtype))[-1].sigmoid()[0, 0].float().cpu().numpy()
            io.submit(save, p, wh, prob)
    print(f"masks {len(files)}: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
