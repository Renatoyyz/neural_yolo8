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

## Configuração do Ambiente Virtual

Para garantir que todas as dependências sejam instaladas corretamente e evitar conflitos com outras bibliotecas, é recomendável criar um ambiente virtual. Siga os passos abaixo para configurar um ambiente virtual e instalar as dependências a partir de um arquivo `requirements.txt`.

### Passo 1: Criação do Ambiente Virtual

Primeiro, crie um ambiente virtual utilizando o `venv`:

```bash
python -m venv venv
```

### Passo 2: Ativação do Ambiente Virtual

Ative o ambiente virtual. O comando para ativar o ambiente virtual varia conforme o sistema operacional:

- **Windows**:
    ```bash
    .\venv\Scripts\activate
    ```

- **macOS e Linux**:
    ```bash
    source venv/bin/activate
    ```

### Passo 3: Instalação das Dependências

Com o ambiente virtual ativado, instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Passo 4: Desativação do Ambiente Virtual

Após terminar de usar o ambiente virtual, você pode desativá-lo com o comando:

```bash
deactivate
```

### Exemplo de Arquivo `requirements.txt`

Aqui está um exemplo de como o arquivo `requirements.txt` pode ser estruturado:

```
opencv-python
torch
matplotlib
ultralytics
```

Certifique-se de que o arquivo `requirements.txt` esteja no mesmo diretório onde você executa o comando `pip install -r requirements.txt`.

## Uso

### YOLOPredictor

A classe `YOLOPredictor` é responsável por carregar o modelo YOLO e realizar predições em imagens.

### Download dos Modelos

Para baixar os modelos do YOLOv8, você pode utilizar o comando `wget` (disponível em sistemas Linux e macOS) ou fazer o download diretamente do repositório oficial. Aqui estão exemplos de como baixar os modelos YOLOv8m, YOLOv8m-seg, YOLOv8n, YOLOv8x e YOLOv8x-seg:

```bash
# YOLOv8m
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8m.pt -O yolov8m.pt

# YOLOv8m-seg
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8m-seg.pt -O yolov8m-seg.pt

# YOLOv8n
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8n.pt -O yolov8n.pt

# YOLOv8x
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8x.pt -O yolov8x.pt

# YOLOv8x-seg
wget https://github.com/ultralytics/yolov8/releases/download/v8.0/yolov8x-seg.pt -O yolov8x-seg.pt
```

Para usuários do Windows, você pode baixar os modelos diretamente do navegador acessando os links fornecidos e salvando os arquivos no diretório apropriado para que possam ser carregados pelo `YOLOPredictor`.

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
