import time
import numpy as np
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import funcoes_desenho
from sklearn.linear_model import RANSACRegressor

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

    def predict(self, img_path, conf=0.3, classes=[39, 40, 41], resize_dim=(640, 640), show_image=True, print_results=True):
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, resize_dim)

        start = time.time()
        resultados = self.model.predict(source=img_resized, conf=conf, device=self.device, classes=classes)
        end = time.time()

        print(f"Tempo total de predição: {(end - start) * 1000:.2f} ms")

        resultado_imagem = funcoes_desenho.desenha_caixas(img_resized.copy(), resultados[0].boxes.data)
        if show_image:
            self.mostrar(resultado_imagem)
        if print_results:
            print(resultados)

            for r in resultados:
                for box in r.boxes.data:
                    x1, y1, x2, y2 = box[:4]
                    print(f"Bounding box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        return resultados, img_resized
    
    def get_bounding_box_info(self, resultados, img_resized, box_index=0):
        if not resultados or len(resultados[0].boxes.data) == 0:
            raise ValueError("No bounding boxes found in the prediction results.")

        box = resultados[0].boxes.data[box_index]
        x1, y1, x2, y2 = map(int, box[:4])

        # Extrair a região da garrafa
        bottle_region = img_resized[y1:y2, x1:x2]

        # Visualizar a imagem extraída
        cv2.imshow('Extracted Region', bottle_region)
        cv2.waitKey(0)

        # Converter para escala de cinza
        gray = cv2.cvtColor(bottle_region, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro de suavização para reduzir o ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Ajustar o contraste da imagem
        alpha = 2.0  # Contraste
        beta = 50    # Brilho
        adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

        # Visualizar a imagem ajustada
        cv2.imshow('Adjusted Image', adjusted)
        cv2.waitKey(0)

        # Aplicar detecção de bordas
        edges = cv2.Canny(adjusted, 50, 150)

        # Visualizar as bordas detectadas
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)

        # Encontrar os pontos de borda
        points = cv2.findNonZero(edges)

        if points is None:
            raise ValueError("No edge points found in the bottle region.")

        # Converter os pontos para o formato adequado para o RANSAC
        points = points.reshape(-1, 2)
        X = points[:, 0].reshape(-1, 1)  # Coordenadas x
        y = points[:, 1]  # Coordenadas y

        # Ajustar uma linha de regressão linear usando RANSAC
        ransac = RANSACRegressor()
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_

        # Calcular a inclinação da linha ajustada
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        angle = np.arctan(slope) * 180 / np.pi

        # Ajustar o ângulo para o intervalo [0, 360]
        if angle < 0:
            angle += 360

        # Visualizar a linha de regressão linear
        line_x = np.array([0, bottle_region.shape[1]])
        line_y = slope * line_x + intercept
        line_y = line_y.astype(int)

        # Desenhar a linha de regressão na imagem
        bottle_region_with_line = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.line(bottle_region_with_line, (line_x[0], line_y[0]), (line_x[1], line_y[1]), (0, 255, 0), 2)

        # Visualizar a imagem com a linha de regressão
        cv2.imshow('Regression Line', bottle_region_with_line)
        cv2.waitKey(0)

        # Cortar a imagem simetricamente ao longo da linha de regressão
        height, width = bottle_region.shape[:2]
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.line(mask, (line_x[0], line_y[0]), (line_x[1], line_y[1]), 255, 1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        top_half = cv2.bitwise_and(bottle_region_with_line, mask)
        bottom_half = cv2.bitwise_and(bottle_region_with_line, cv2.bitwise_not(mask))

        # Visualizar as metades cortadas
        cv2.imshow('Top Half', top_half)
        cv2.imshow('Bottom Half', bottom_half)
        cv2.waitKey(0)

        # Calcular o centro do bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return {
            "center_x": center_x,
            "center_y": center_y,
            "orientation": angle
        }

# Exemplo de uso da classe
if __name__ == "__main__":
    predictor = YOLOPredictor()

    # Primeira predição
    img_path = 'YOLO_recursos/Atualização YOLOv8/fotos_teste/potes_virado1.jpeg'
    result, img_resized = predictor.predict(img_path=img_path, conf=0.2, classes=[39,40,41], show_image=True, print_results=True)

    for i in range(len(result[0].boxes.data)):
        box_info = predictor.get_bounding_box_info(result, img_resized, box_index=i)
        print(f"Box {i} info: {box_info}")

    # Segunda predição
    predictor.predict(img_path='YOLO_recursos/Atualização YOLOv8/fotos_teste/potes_virado1.jpeg', show_image=False, print_results=False)