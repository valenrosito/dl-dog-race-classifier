import os
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
from GradioStage3 import model as model_stage3, clasificar_recorte

# ========== CONFIGURACIÓN ==========
IMG_DIR   = "test-images"
OUT_DIR   = "resnet-annotations"
COCO_JSON = os.path.join(OUT_DIR, "coco_annotations.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Genera id2raza a partir de los nombres de los directorios en dog-images/test

breed_dirs = sorted([d for d in os.listdir("dog-images/test") if os.path.isdir(os.path.join("dog-images/test", d))])
id2raza = {i: breed for i, breed in enumerate(breed_dirs)}
raza2id = {v: k for k, v in id2raza.items()}

# ========== PREPARAR ESTRUCTURAS COCO ==========
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}
# categorías COCO
for cid, name in id2raza.items():
    coco["categories"].append({
        "id": cid,
        "name": name,
        "supercategory": "dog_breed"
    })

annotation_id = 0

yolo = model_stage3 

# ========== BUCLE DE ANOTACIÓN ==========
for image_id, fname in enumerate(sorted(os.listdir(IMG_DIR))):
    if not fname.lower().endswith(('.jpg', '.png')): continue
    img_path = os.path.join(IMG_DIR, fname)
    img_pil  = Image.open(img_path).convert("RGB")
    w, h     = img_pil.size
    img_np   = np.array(img_pil)

    # añadir info imagen COCO
    coco["images"].append({
        "id": image_id,
        "file_name": fname,
        "width": w,
        "height": h
    })

    # detección YOLO
    res = yolo.predict(source=img_np, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)

    # filtrar perros (class_id==16)
    mask = classes == 16
    boxes = boxes[mask]

    # preparar archivo YOLOv5 .txt
    txt_lines = []

    for box in boxes:
        x1, y1, x2, y2 = box
        # normalizar bbox
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        # recortar y clasificar
        crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
        breed = clasificar_recorte(crop)
        cid   = raza2id[breed]

        # línea YOLOv5: class cx cy w h
        txt_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # añadir anotación COCO
        area = float((x2 - x1) * (y2 - y1))
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cid,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "area": area,
            "iscrowd": 0
        })
        annotation_id += 1

    # guardar .txt YOLOv5
    txt_path = os.path.join(OUT_DIR, os.path.splitext(fname)[0] + ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

# ========== GUARDAR ANOTACIONES COCO ==========
with open(COCO_JSON, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ Anotaciones generadas en {OUT_DIR}/ (YOLO .txt + COCO .json)")