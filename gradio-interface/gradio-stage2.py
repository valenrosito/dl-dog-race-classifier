import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 as TFResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as tf_preprocess
import faiss
import numpy as np
import gradio as gr
from PIL import Image

# 1) Definición del modelo custom
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,224,224)
            flat_dim = self.conv(dummy).view(1,-1).shape[1]
        self.flat_dim = flat_dim     # <<-- almacenar dimensión
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    def extract(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)

# 2) Preparacion device, transform y datasets
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

tf.config.set_visible_devices([], 'GPU')


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
data_root = "./dog-images"
train_ds = datasets.ImageFolder(f"{data_root}/train", transform=transform)
all_paths = [p for p,_ in train_ds.imgs]  # para mostrar luego
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 3) Instanciado de modelos
num_classes = len(train_ds.classes)
# ResNet18
r18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
r18_fc_in = r18.fc.in_features
r18.fc = nn.Linear(r18_fc_in, num_classes)
# ResNet50
tf_r50 = TFResNet50(weights='imagenet', include_top=False, pooling='avg')
tf_r50.trainable = False
tf50_dim = 2048
# Custom CNN
custom = SimpleCNN(num_classes)

for m in (r18, custom):
    m.to(device)

criterion = nn.CrossEntropyLoss()
opt_r18 = torch.optim.Adam(r18.parameters(), lr=1e-4)
opt_c  = torch.optim.Adam(custom.parameters(), lr=1e-4)

# 4) Entrenamiento de los modelos
def train_one_epoch(model, opt):
    model.train()
    for imgs, labels in tqdm(train_loader, desc=f"Entrenando {model.__class__.__name__}", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        opt.step()

print("Entrenando modelo ResNet18...")
for epoch in range(5):
    train_one_epoch(r18, opt_r18)
print("Omitiendo entrenamiento de ResNet50 (uso de preentrenado TF)")
print("Entrenando modelo Custom CNN...")
for epoch in range(5):
    train_one_epoch(custom, opt_c)

# 5) Construccion de extractores de features (sin la capa FC)
feat_r18 = nn.Sequential(*list(r18.children())[:-1]).to(device)  # sale (N,512,1,1)

def extract_tf_pil(img_pil):
    arr = np.array(img_pil.resize((224,224)))
    arr = tf_preprocess(arr)          
    feat = tf_r50(np.expand_dims(arr,0))  # (1,2048)
    return feat.numpy()   


# 6) Generacion matrix de features y FAISS
def build_index(model_feat, extractor, dim):
    if model_feat is None:
        arrs = []
        for img_path, _ in tqdm(train_ds.imgs, desc="Preparando lote TF", unit="img"):
            img = image.load_img(img_path, target_size=(224,224))
            x = image.img_to_array(img)
            x = tf_preprocess(x)
            arrs.append(x)
        X_in = np.stack(arrs, axis=0)  # (N,224,224,3)
        # Predict en CPU para evitar errores de copia GPU
        with tf.device('/CPU:0'):
            feats = tf_r50.predict(X_in, batch_size=32, verbose=1)  # (N,2048)
        X = feats.astype('float32')
    else:
        feats = []
        for img_path, _ in tqdm(train_ds.imgs, desc=f"Indexando con {extractor.__class__.__name__}", unit="img"):
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = extractor(x)
            feats.append(f.view(1,-1).cpu().numpy())
        X = np.vstack(feats).astype('float32')
    idx = faiss.IndexFlatL2(dim)
    idx.add(X)
    return idx

print("Construyendo índice FAISS para ResNet18...")
idx_r18 = build_index(r18, feat_r18, r18_fc_in)
print("Construyendo índice FAISS para ResNet50...") 
idx_r50 = build_index(None, extract_tf_pil, tf50_dim)
print("Construyendo índice FAISS para Custom CNN...")
idx_c = build_index(custom, custom.extract, custom.flat_dim)

# 7) Función de búsqueda
def extraer_raza(path):
    return os.path.basename(os.path.dirname(path))


def buscar(img, modelo, k=10):
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img).convert("RGB")
    else:
        img_pil = img.convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)

    if modelo=="ResNet18":
        with torch.no_grad():
            f = feat_r18(x).view(1,-1).cpu().numpy().astype('float32')
        D, I = idx_r18.search(f, k)
    elif modelo=="ResNet50":
        f = extract_tf_pil(img_pil).astype('float32')
        D, I = idx_r50.search(f, k)
    else:
        with torch.no_grad():
            f = custom.extract(x).cpu().numpy().astype('float32')
        D, I = idx_c.search(f, k)

    # paths y galerías
    similares = [train_ds.imgs[i][0] for i in I[0]]
    razas = [extraer_raza(p) for p in similares]
    raza_predicha = Counter(razas).most_common(1)[0][0]
    imgs_resultado = [Image.open(p).convert("RGB") for p in similares]
    return imgs_resultado, f"Raza predicha: {raza_predicha}"



# 8) Interfaz Gradio
iface = gr.Interface(
    fn=buscar,
    inputs=[
        gr.Image(type="numpy", label="Imagen consulta"),
        gr.Dropdown(["ResNet18","ResNet50","CustomCNN"], label="Modelo")
    ],
    outputs=[
        gr.Gallery(label="Top similares", columns=5),
        gr.Text(label="Predicción de raza")
    ],
    title="Búsqueda por similitud con FAISS y predictor de raza",
    description="Elige el modelo para extraer características y votar la raza."
)

if __name__=="__main__":
    iface.launch()