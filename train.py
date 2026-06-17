"""
Fine-tune YOLOv8 on the pothole detection dataset.

Uso:
    python train.py
    python train.py --epochs 100 --model yolov8s.pt
    python train.py --epochs 50 --device 0   # GPU
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina YOLOv8 para detecção de buracos em vias"
    )
    parser.add_argument(
        "--data", default="dataset/data.yaml",
        help="Caminho para o arquivo data.yaml do dataset",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Modelo base: yolov8n.pt (rápido) / yolov8s.pt / yolov8m.pt",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Número de épocas de treinamento (padrão: 50)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamanho da imagem de entrada (padrão: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Tamanho do batch (reduza se faltar memória, padrão: 8)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Dispositivo: 'cpu' ou '0' para primeira GPU",
    )
    parser.add_argument(
        "--name", default="pothole_v1",
        help="Nome do experimento (salvo em runs/train/)",
    )
    return parser.parse_args()


def validate_dataset(data_yaml: Path) -> bool:
    if not data_yaml.exists():
        print(f"\n[ERRO] Arquivo não encontrado: {data_yaml}")
        print("       Baixe o dataset do Kaggle e coloque em dataset/")
        print("       Link: https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8")
        return False
    return True


def copy_best_model(run_name: str) -> Path | None:
    best = Path(f"runs/train/{run_name}/weights/best.pt")
    if not best.exists():
        return None
    dest = Path("models/pothole_best.pt")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(best, dest)
    return dest


def main():
    args = parse_args()
    data_yaml = Path(args.data)

    if not validate_dataset(data_yaml):
        return

    print(f"\n{'='*55}")
    print(f"  PotHole Detector — Treinamento YOLOv8")
    print(f"{'='*55}")
    print(f"  Modelo base  : {args.model}")
    print(f"  Épocas       : {args.epochs}")
    print(f"  Batch        : {args.batch}")
    print(f"  Imagem (px)  : {args.imgsz}")
    print(f"  Dispositivo  : {args.device}")
    print(f"  Dataset      : {data_yaml}")
    print(f"{'='*55}\n")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/train",
        patience=15,        # early stopping após 15 épocas sem melhora
        save=True,
        save_period=10,     # salva checkpoint a cada 10 épocas
        verbose=True,
        plots=True,         # gera gráficos de métricas
        val=True,
    )

    dest = copy_best_model(args.name)
    if dest:
        print(f"\n[OK] Melhor modelo salvo em: {dest}")
        print(f"     Carregue-o na GUI com o botão 'Carregar Modelo'.")
    else:
        print("\n[AVISO] best.pt não encontrado. Verifique runs/train/")

    print(f"\n[INFO] Resultados e gráficos em: runs/train/{args.name}/\n")


if __name__ == "__main__":
    main()
