# YOLOv8 Predictor

Este projeto utiliza o modelo YOLOv8 para realizar predições em imagens. Abaixo estão as instruções para instalação das dependências, uso e operação do código.

## Instalação de Dependências

Para executar este projeto, você precisará instalar as seguintes dependências:

- Python 3.8 ou superior
- OpenCV
- Torch
- Matplotlib
- Ultralytics YOLO

Você pode instalar todas as dependências necessárias utilizando o `pip`:

```bash
pip install opencv-python torch matplotlib ultralytics
```

## Uso

### YOLOPredictor

A classe `YOLOPredictor` é responsável por carregar o modelo YOLO e realizar predições em imagens.

#### Inicialização

Para inicializar a classe, você pode especificar o caminho do modelo e o dispositivo (CPU ou GPU) a ser utilizado:

```python
predictor = YOLOPredictor(model_path='yolov8m-seg.pt', device='cuda')
```

#### Predição

Para realizar uma predição, utilize o método `predict`:

```python
resultados = predictor.predict(
    img_path='caminho/para/sua/imagem.jpg',
    conf=0.2,
    classes=[39, 40, 41],
    resize_dim=(320, 320),
    show_image=True,
    print_results=True
)
```

- `img_path`: Caminho para a imagem a ser predita.
- `conf`: Confiança mínima para considerar uma detecção.
- `classes`: Lista de classes a serem detectadas.
- `resize_dim`: Dimensões para redimensionar a imagem.
- `show_image`: Se `True`, mostra a imagem com as detecções.
- `print_results`: Se `True`, imprime os resultados no console.

### Funções de Desenho

O arquivo `funcoes_desenho.py` contém funções auxiliares para desenhar caixas delimitadoras e rótulos nas imagens.

#### `desenha_caixas`

Esta função desenha caixas delimitadoras sobre a imagem:

```python
from funcoes_desenho import desenha_caixas

imagem_com_caixas = desenha_caixas(image, boxes, labels=[], colors=[], score=True, conf=None)
```

- `image`: Imagem sobre a qual desenhar as caixas.
- `boxes`: Caixas delimitadoras a serem desenhadas.
- `labels`: Lista de rótulos das classes.
- `colors`: Lista de cores para cada classe.
- `score`: Se `True`, adiciona a confiança ao lado do rótulo.
- `conf`: Confiança mínima para desenhar a caixa.

## Exemplo de Uso

```python
if __name__ == "__main__":
    predictor = YOLOPredictor()

    # Primeira predição
    predictor.predict(img_path='YOLO_recursos/Atualização YOLOv8/fotos_teste/potes.jpeg', show_image=False, print_results=False)

    # Segunda predição
    predictor.predict(img_path='YOLO_recursos/Atualização YOLOv8/fotos_teste/potes2.jpeg', show_image=False, print_results=False)
```

Este exemplo inicializa o `YOLOPredictor` e realiza duas predições em imagens diferentes.
