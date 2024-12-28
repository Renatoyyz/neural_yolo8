import time
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import funcoes_desenho

class YOLOPredictor:
    def __init__(self, model_path='yolov8m-seg.pt', device=None):
        # Verifica se a GPU está disponível
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")

        # Carregar o modelo YOLO
        self.model = YOLO(model_path)
        if self.model is None:
            raise ValueError("Failed to load the YOLO model.")
        self.model.to(self.device)

    def mostrar(self, img):
        fig = plt.gcf()
        fig.set_size_inches(16, 10)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def predict(self, img_path, conf=0.2, classes=[39, 40, 41], resize_dim=(320, 320), show_image=True, print_results=True):
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, resize_dim)

        start = time.time()
        resultados = self.model.predict(source=img_resized, conf=conf, device=self.device, classes=classes)
        end = time.time()

        print(f"Tempo total de predição: {(end - start) * 1000:.2f} ms")

        resultado_imagem = funcoes_desenho.desenha_caixas(img_resized, resultados[0].boxes.data)
        if show_image:
            self.mostrar(resultado_imagem)
        if print_results:
            print(resultados)

            for r in resultados:
                for box in r.boxes.data:
                    x1, y1, x2, y2 = box[:4]
                    print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        return resultados

# Exemplo de uso da classe
if __name__ == "__main__":
    predictor = YOLOPredictor()

    # Primeira predição
    predictor.predict( img_path= 'YOLO_recursos/Atualização YOLOv8/fotos_teste/potes.jpeg', show_image=False, print_results=False)

    # Segunda predição
    predictor.predict( img_path='YOLO_recursos/Atualização YOLOv8/fotos_teste/potes2.jpeg', show_image=False, print_results=False)