# PotHole Detector

Sistema de detecção de irregularidades em vias públicas (buracos) utilizando YOLOv8 e OpenCV.

Desenvolvido como trabalho prático da disciplina **CCC309 – Processamento de Imagens e Visão Computacional** (UPF – 2026/1).

## Demo

> (https://drive.google.com/drive/folders/1ImgzYjmPB0KxtHDnxikbiwliH9S7kwdz?usp=sharing)

## Funcionalidades

- Detecção de buracos em imagens, vídeos e câmera ao vivo (webcam)
- Interface gráfica com PyQt5 e tema escuro
- Pipeline de pré-processamento com OpenCV:
  - Suavização Gaussiana (redução de ruído)
  - Realce de contraste adaptativo (CLAHE)
  - Operações morfológicas (fechamento)
  - Detecção de bordas (Canny)
  - Conversão para escala de cinza
- Classificação de severidade: Severo / Moderado / Leve
- Estatísticas em tempo real (contagem, confiança média)
- Exportação do frame anotado como imagem

## Instalação

### Requisitos

- Python 3.10+
- pip

### Configuração

```bash
# Clonar o repositório
git clone https://github.com/BernardoBones/deteccao_buracos_em_ruas
cd deteccao_buracos_em_ruas

# Criar ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

## Dataset

Baixe o dataset do Kaggle (requer conta gratuita):

- [anggadwisunarto/potholes-detection-yolov8](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)

Extraia na pasta `dataset/` com a seguinte estrutura:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Treinamento

```bash
python train.py --epochs 50 --model yolov8n.pt --device cpu
```

O melhor modelo será salvo automaticamente em `models/pothole_best.pt`.

## Uso

```bash
python main.py
```

1. Clique em **Abrir Imagem**, **Abrir Vídeo** ou **Webcam**
2. Ajuste o **Limiar de Confiança** conforme necessário
3. Ative os **Filtros de Pré-processamento** desejados
4. Carregue um modelo treinado via **Carregar Modelo** (opcional — usa yolov8n por padrão)
5. Salve os resultados com **Salvar Imagem**

## Estrutura do Projeto

```
pothole-detector/
├── app/
│   ├── main_window.py      # Interface gráfica PyQt5
│   ├── detector.py         # Detecção YOLOv8
│   ├── image_processor.py  # Pipeline de filtros OpenCV
│   └── video_thread.py     # Thread de captura de vídeo
├── models/                 # Modelos treinados (.pt) — não versionado
├── dataset/                # Dataset — não versionado
├── train.py                # Script de fine-tuning
├── main.py                 # Ponto de entrada
└── requirements.txt
```

## Integrantes

- Bernardo Baroni Bones - 192298
- Pedro Antônio da Silva - 194828

## Professor

Prof. Dr. Rafael Rieder — rieder@upf.br
