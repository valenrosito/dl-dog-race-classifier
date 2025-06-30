import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from ultralytics import YOLO
from GradioStage3 import clasificar_recorte as clasificar_stage3, model as model_stage3

# ========== PARTE 1: Preparación del dataset ==========
train_dir = "dog-images/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))


# ========== PARTE 2: Crear y guardar modelo en SavedModel ==========
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

clf_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(9, activation='softmax')  # 9 clases de raza
])
clf_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

saved_model_dir = "resnet50_dogs_saved"
clf_model.export(saved_model_dir)
print(f"✅ Modelo guardado en SavedModel en {saved_model_dir}")


# ========== PARTE 3: Conversión a TFLite (float16) ==========
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16 = converter.convert()
tflite_model_path = "resnet50_fp16.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_fp16)
print(f"✅ Conversión a TFLite FP16 exitosa: {tflite_model_path}")


# ========== PARTE 4: Medición tiempo de inferencia ==========
def medir_tiempo_tf_model(model, dataset, num_batches=10):
    start = time.time()
    for batch_images, _ in dataset.take(num_batches):
        _ = model(batch_images, training=False)
    return time.time() - start

def medir_tiempo_tflite_model(tflite_path, dataset, num_batches=10):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']

    start = time.time()
    for batch_images, _ in dataset.take(num_batches):
        batch_np = batch_images.numpy()
        for img in batch_np:
            interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
            interpreter.invoke()
    return time.time() - start

print("\n⏱️ Tiempos de inferencia (10 batches):")
tiempo_tf = medir_tiempo_tf_model(clf_model, train_dataset)
tiempo_tflite = medir_tiempo_tflite_model(tflite_model_path, train_dataset)
print(f"🔹 ResNet50 TensorFlow TF32: {tiempo_tf:.2f}s")
print(f"🔸 ResNet50 TFLite FP16:      {tiempo_tflite:.2f}s")


# ========== PARTE 5: Evaluación pipeline detección + clasificación ==========
# Utilidades comunes
id2raza = {
    0: 'Beagle', 1: 'Japanese Spaniel', 2: 'Labrador',
    3: 'Dobberman', 4: 'Yorkie', 5: 'Pug',
    6: 'Golden Retriever', 7: 'German Sheperd', 8: 'Shiba Inu'
}
raza2id = {v: k for k, v in id2raza.items()}

def etiquetas_texto_a_id(lista_texto, mapa):
    return [mapa.get(t, -1) for t in lista_texto]

def cargar_anotaciones_yolo(path_txt, img_w, img_h):
    boxes, labels = [], []
    with open(path_txt, 'r') as f:
        for line in f:
            cid, x_c, y_c, w, h = map(float, line.strip().split())
            x1 = (x_c - w/2) * img_w
            y1 = (y_c - h/2) * img_h
            x2 = (x_c + w/2) * img_w
            y2 = (y_c + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cid))
    return boxes, labels

def cargar_ground_truth_yolo(img_folder, ann_folder):
    gt_boxes, gt_labels = [], []
    for name in sorted(os.listdir(img_folder)):
        if not name.lower().endswith(('.jpg', '.png')): continue
        img = Image.open(os.path.join(img_folder, name))
        w, h = img.size
        txt = os.path.join(ann_folder, os.path.splitext(name)[0] + '.txt')
        b, l = cargar_anotaciones_yolo(txt, w, h)
        gt_boxes.append(b); gt_labels.append(l)
    return gt_boxes, gt_labels

def iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    A1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    A2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter/(A1+A2-inter) if (A1+A2-inter)>0 else 0

def match_gt_pred(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh=0.5):
    y_true, y_pred = [], []
    for gbs, gls, pbs, pls in zip(gt_boxes, gt_labels, pred_boxes, pred_labels):
        used, n = set(), len(pbs)
        for gi, gb in enumerate(gbs):
            best_i, best_pi = 0, None
            for pi in range(n):
                if pi in used: continue
                i = iou(gb, pbs[pi])
                if i>best_i: best_i, best_pi = i, pi
            if best_pi is not None and best_i>=iou_thresh:
                y_true.append(gls[gi]); y_pred.append(pls[best_pi]); used.add(best_pi)
            else:
                y_true.append(gls[gi]); y_pred.append(-1)
    return y_true, y_pred

# Clasificador TFLite FP16
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
inp_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

def clasificar_tflite(crop_img):
    img = crop_img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)/255.0
    interpreter.set_tensor(inp_idx, arr[np.newaxis,...])
    interpreter.invoke()
    out = interpreter.get_tensor(out_idx)[0]
    return np.argmax(out)

# Pipeline detección + clasificación
def detectar_y_clasificar_batch(img_folder, clasificador_fn):
    model_yolo = YOLO("yolov8n.pt")
    pb, pl, ps = [], [], []
    for name in sorted(os.listdir(img_folder)):
        if not name.lower().endswith(('.jpg','.png')): continue
        img = np.array(Image.open(os.path.join(img_folder, name)).convert("RGB"))
        res = model_yolo.predict(source=img, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        cls   = res.boxes.cls.cpu().numpy().astype(int)
        conf  = res.boxes.conf.cpu().numpy()
        mask = cls==16
        img_pil = Image.fromarray(img)
        b_img,l_img,s_img = [],[],[]
        for i,box in enumerate(boxes[mask]):
            x1,y1,x2,y2 = [int(v) for v in box]
            crop = img_pil.crop((x1,y1,x2,y2))
            pred = clasificador_fn(crop)
            b_img.append([x1,y1,x2,y2]); l_img.append(pred); s_img.append(float(conf[mask][i]))
        pb.append(b_img); pl.append(l_img); ps.append(s_img)
    return pb, pl, ps

def evaluar_modelo(nombre, clas_fn, gt_boxes, gt_labels, img_dir):
    pb, pl, _ = detectar_y_clasificar_batch(img_dir, clas_fn)
    if pl and isinstance(pl[0][0], str):
        pl = [etiquetas_texto_a_id(lbls, raza2id) for lbls in pl]
    y_true, y_pred = match_gt_pred(gt_boxes, gt_labels, pb, pl)
    mask = [p!=-1 for p in y_pred]
    yt = [yt for yt,m in zip(y_true,mask) if m]
    yp = [yp for yp,m in zip(y_pred,mask) if m]
    p = precision_score(yt, yp, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f = f1_score(yt, yp, average='macro', zero_division=0)
    print(f"\n{nombre}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")
    return p,r,f

# ======== MAIN ========
if __name__ == "__main__":
    img_dir = "test-images/"
    ann_dir = "test-annotations/"

    gt_boxes, gt_labels = cargar_ground_truth_yolo(img_dir, ann_dir)

    print("\n➡️ Evaluando clasificación en detecciones:")
    evaluar_modelo("🔹 ResNet50 TensorFlow TF32", clasificar_stage3, gt_boxes, gt_labels, img_dir)
    evaluar_modelo("🔸 ResNet50 TFLite FP16", clasificar_tflite, gt_boxes, gt_labels, img_dir)
