import torch
from ultralytics import YOLO
import gradio as gr
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from tensorflow.keras.applications import ResNet50 as TFResNet50
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as tf_preprocess
from tensorflow.keras.preprocessing import image
import faiss
from tqdm import tqdm
from torchvision import datasets, transforms

# Configuración de dispositivo
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Usando dispositivo YOLO: {device}")

# Dataset y transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
data_root = "./dog-images"
train_ds = datasets.ImageFolder(f"{data_root}/train", transform=transform)

# Modelo ResNet50 de Keras
tf_r50 = TFResNet50(weights='imagenet', include_top=False, pooling='avg')
tf_r50.trainable = False
tf50_dim = 2048

# Indexación FAISS para ResNet50
def build_index():
    arrs = []
    for img_path, _ in tqdm(train_ds.imgs, desc="Preparando lote TF", unit="img"):
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = tf_preprocess(x)
        arrs.append(x)
    X_in = np.stack(arrs, axis=0)  # (N,224,224,3)
    with tf.device('/CPU:0'):
        feats = tf_r50.predict(X_in, batch_size=32, verbose=1)  # (N,2048)
    X = feats.astype('float32')
    idx = faiss.IndexFlatL2(tf50_dim)
    idx.add(X)
    return idx

idx_r50 = build_index()

def extract_tf_pil(img_pil):
    arr = np.array(img_pil.resize((224,224)))
    arr = tf_preprocess(arr)
    feat = tf_r50(np.expand_dims(arr,0))
    return feat.numpy()

def extraer_raza(path):
    return os.path.basename(os.path.dirname(path))

# Clasificación de recortes usando ResNet50 + FAISS
def clasificar_recorte(crop_pil, k=10):
    f = extract_tf_pil(crop_pil).astype('float32')
    D, I = idx_r50.search(f, k)
    similares = [train_ds.imgs[i][0] for i in I[0]]
    razas = [extraer_raza(p) for p in similares]
    return Counter(razas).most_common(1)[0][0]

# Carga modelo YOLOv8
model = YOLO('yolov8n.pt')
model.to(device)

# Detección y clasificación
def detectar_y_clasificar(img):
    img_pil = Image.fromarray(img).convert("RGB")
    results = model(img_pil)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    dog_boxes = boxes[classes == 16]  # class_id 16 = dog

    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("Helvetica", 40)
    except:
        font = ImageFont.load_default()

    for box in dog_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img_pil.crop((x1, y1, x2, y2))
        raza = clasificar_recorte(crop)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Sombra: dibuja el texto en negro ligeramente desplazado
        shadow_offset = 2
        draw.text((x1 + shadow_offset, y1 - 10 + shadow_offset), raza, fill="black", font=font)
        # Texto principal
        draw.text((x1, y1 - 10), raza, fill="white", font=font)
    return img_pil

# Interfaz Gradio
iface = gr.Interface(
    fn=detectar_y_clasificar,
    inputs=gr.Image(type="numpy", label="Imagen de entrada"),
    outputs=gr.Image(type="pil", label="Detección y raza"),
    title="Detección y clasificación de razas de perros (ResNet50)",
    description="Carga una imagen compleja, detecta perros y etiqueta su raza usando ResNet50."
)

if __name__ == "__main__":
    iface.launch()