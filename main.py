import time
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import funcoes_desenho

# Verifica se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

model_x = YOLO('yolov8m.pt')
if model_x is None:
  raise ValueError("Failed to load the YOLO model.")
model_x.to(device)

def mostrar(img):
  fig = plt.gcf()
  fig.set_size_inches(16, 10)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

start = time.time()

caminh_img = 'YOLO_recursos/Atualização YOLOv8/fotos_teste/potes2.jpeg'
img = cv2.imread(caminh_img)

# Reduz o tamanho da imagem para acelerar a predição
img_resized = cv2.resize(img, (640, 640))

resultados = model_x.predict(source=img_resized, conf=0.2, device=device, classes=[39,40,41])  # Classe 16 é geralmente 'dog' no COCO dataset

end = time.time()
print(f"Tempo total de predição: {(end - start) * 1000:.2f} ms")

resultado_imagem = funcoes_desenho.desenha_caixas(img_resized, resultados[0].boxes.data)

mostrar(resultado_imagem)
print(resultados)

for r in resultados:
  for box in r.boxes.data:
    x1, y1, x2, y2 = box[:4]
    print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
