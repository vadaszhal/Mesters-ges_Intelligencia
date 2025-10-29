# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install transformers pillow

from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    ViTImageProcessor, ViTForImageClassification,
    BlipProcessor, BlipForConditionalGeneration
)
from PIL import Image
import torch

# figyelmezetetés eltüntetése
import warnings
from transformers.utils import logging as hf_logging

# PyTorch figyelmeztetések elnyomása
warnings.filterwarnings("ignore", category=UserWarning)

# Transformers (Hugging Face) figyelmeztetések elnyomása
hf_logging.set_verbosity_error()

# figyelmeztetés vége

IMAGE_PATH = "images/5.picture.jpg"
DET_SCORE_THR = 0.90
TOPK = 3
USE_FP16 = True  # ha gond van (pl. pascal kártyán), állítsd False-ra

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple
        return torch.device("mps")
    else:
        return torch.device("cpu")

def main():
    device = get_device()
    print(f"Eszköz: {device}")

    # --- Modellek betöltése + eval ---
    detr_proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()

    vit_proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device).eval()

    # BLIP-et tedd CPU-ra, ha szűk a VRAM (GTX 1050 esetén hasznos lehet):
    blip_device = device if (device.type != "cuda") else device  # vagy torch.device("cpu")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(blip_device).eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    # Autocast csak CUDA-n érdemes
    use_autocast = (device.type == "cuda") and USE_FP16
    amp_ctx = torch.cuda.amp.autocast if use_autocast else torch.cpu.amp.autocast  # cpu.amp 2.4+-tól elérhető; ha nincs, nem gond.

    print("\n🔍 1. Objektumfelismerés (DETR)...")
    with torch.inference_mode():
        inputs = detr_proc(images=image, return_tensors="pt").to(device)
        with (amp_ctx() if use_autocast else torch.no_grad()):
            outputs = detr(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = detr_proc.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

        # Szűrés és rendezés pontosság szerint
        dets = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            s = score.item()
            if s >= DET_SCORE_THR:
                dets.append({
                    "score": s,
                    "label": detr.config.id2label[label.item()],
                    "box": box.detach().cpu().tolist()
                })
        dets.sort(key=lambda x: x["score"], reverse=True)

        if dets:
            for d in dets:
                print(f' - {d["label"]}: {d["score"]:.2f}, Box: {d["box"]}')
        else:
            print(" - Nincs 0.90 feletti találat (próbálj alacsonyabb küszöböt, pl. 0.7).")

    print("\n🖼️ 2. Képosztályozás (ViT, top-k a fő tárgy kivágásán)...")
    # Ha van detektálás, a legjobb dobozt vágjuk ki; különben teljes képet osztályozunk.
    if dets:
        xmin, ymin, xmax, ymax = dets[0]["box"]
        crop = image.crop((xmin, ymin, xmax, ymax))
        cls_img = crop
    else:
        cls_img = image

    with torch.inference_mode():
        cls_inputs = vit_proc(images=cls_img, return_tensors="pt").to(device)
        with (amp_ctx() if use_autocast else torch.no_grad()):
            logits = vit(**cls_inputs).logits
        probs = logits.softmax(dim=-1)[0]
        topk = torch.topk(probs, k=min(TOPK, probs.shape[-1]))
        for i in range(topk.indices.numel()):
            idx = topk.indices[i].item()
            p = topk.values[i].item()
            print(f" - Top{i+1}: {vit.config.id2label[idx]} ({p:.2%})")

    print("\n📝 3. Képleírás generálás (BLIP)...")
    with torch.inference_mode():
        blip_inputs = blip_proc(images=image, return_tensors="pt").to(blip_device)
        # BLIP gyakran jobb teljes FP32-ben régebbi GPU-kon; ha kifut a VRAM-ból, tedd CPU-ra.
        out = blip.generate(**blip_inputs, max_new_tokens=30)
        caption = blip_proc.decode(out[0], skip_special_tokens=True)
        print(f" - Képleírás: {caption}")

if __name__ == "__main__":
    main()
