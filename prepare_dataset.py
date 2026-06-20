"""
Prepara o dataset do Kaggle para uso com YOLOv8.

Fontes aceitas (qualquer uma):
  1. Arquivo zip do Kaggle ainda não extraído (ex.: archive.zip)
  2. Pasta já extraída do Kaggle (contém train/, valid/, test/)

Uso:
    python prepare_dataset.py                          # procura automaticamente
    python prepare_dataset.py --source archive.zip
    python prepare_dataset.py --source pasta_extraida/
    python prepare_dataset.py --verify                 # só verifica sem mover nada
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

DATASET_DIR = Path("dataset")
SPLITS = ["train", "valid", "test"]
OPTIONAL_SPLITS = {"test"}
REQUIRED_SUBDIRS = ["images", "labels"]

# Sufixos aceitos como imagem
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# Helpers
def _count(directory: Path, exts: set[str]) -> int:
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in exts)


def _find_split_root(base: Path) -> Path | None:
    """Encontra o diretório que contém train/, valid/ ou test/."""
    if any((base / s).is_dir() for s in SPLITS):
        return base
    for child in base.iterdir():
        if child.is_dir() and any((child / s).is_dir() for s in SPLITS):
            return child
    return None


# Extração do zip
def extract_zip(zip_path: Path, dest: Path) -> Path:
    print(f"[INFO] Extraindo {zip_path.name} → {dest} …")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print(f"[OK]   Extração concluída em {dest}")
    return dest


# Reorganização das pastas
def reorganize(source_root: Path) -> bool:
    """
    Copia train/valid/test de source_root para dataset/.
    Mantém source_root intacto (copia, não move).
    """
    print(f"\n[INFO] Reorganizando dataset de: {source_root}")
    ok = True

    for split in SPLITS:
        src_split = source_root / split
        dst_split = DATASET_DIR / split

        if not src_split.is_dir():
            print(f"[AVISO] Split '{split}' não encontrado em {source_root} — pulando.")
            continue

        for sub in REQUIRED_SUBDIRS:
            src = src_split / sub
            dst = dst_split / sub

            if not src.is_dir():
                print(f"[AVISO] {split}/{sub} não existe na fonte — pulando.")
                continue

            if dst.exists():
                shutil.rmtree(dst)
            print(f"         Copiando {split}/{sub} …", end=" ", flush=True)
            shutil.copytree(src, dst)
            n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
            print(f"{n_files} arquivo(s)")

    return ok


# Verificação
def verify() -> bool:
    print(f"\n{'='*55}")
    print(f"   Verificação do Dataset")
    print(f"{'='*55}")

    all_ok = True

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        imgs_dir = split_dir / "images"
        lbs_dir = split_dir / "labels"

        n_imgs = _count(imgs_dir, IMAGE_EXTS) if imgs_dir.is_dir() else 0
        n_lbs = _count(lbs_dir, {".txt"}) if lbs_dir.is_dir() else 0

        if n_imgs == 0:
            if split in OPTIONAL_SPLITS:
                status = "[INFO]"
                print(f"   {status} {split:<6}: {n_imgs:>5} imagens  |  {n_lbs:>5} labels (Split opcional ausente)")
                continue
            else:
                status = "[ERRO]"
                all_ok = False
        else:
            status = "[OK]  " if n_imgs == n_lbs else "[WARN]"

        print(f"   {status} {split:<6}: {n_imgs:>5} imagens  |  {n_lbs:>5} labels", end="")
        if n_imgs != n_lbs:
            print(f"   ← contagens diferentes!", end="")
            all_ok = False
        print()

    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"\n   [OK]   data.yaml encontrado: {data_yaml}")
    else:
        print(f"\n   [ERRO] data.yaml não encontrado em {data_yaml}")
        all_ok = False

    print(f"\n{'='*55}")
    if all_ok:
        print("   Dataset pronto para treinamento.")
        print("   Execute: python train.py --epochs 50 --batch 8")
    else:
        print("   Corrija os erros acima antes de treinar.")
    print(f"{'='*55}\n")

    return all_ok


# Auto-detecção de fonte
def autodetect_source() -> Path | None:
    """Procura zip ou pasta extraída no diretório atual e em Downloads."""
    candidates: list[Path] = []

    # Zips na raiz do projeto
    candidates += sorted(Path(".").glob("*.zip"))

    # Pasta 'downloads' ou 'Downloads' local
    for dl in [Path("downloads"), Path("Downloads")]:
        if dl.is_dir():
            candidates += sorted(dl.glob("*.zip"))
            candidates += [dl]

    # Pasta já com splits dentro de dataset/ (nada a fazer)
    split_root = _find_split_root(DATASET_DIR)
    if split_root == DATASET_DIR:
        return DATASET_DIR  # já organizado

    # Pastas irmãs que parecem um dataset YOLOv8
    for d in Path(".").iterdir():
        if d.is_dir() and d != DATASET_DIR and _find_split_root(d):
            candidates.append(d)

    return candidates[0] if candidates else None


# Main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara o dataset de potholes para YOLOv8"
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Zip ou pasta extraída do Kaggle (detectado automaticamente se omitido)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Apenas verifica o dataset sem mover arquivos",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verify:
        ok = verify()
        sys.exit(0 if ok else 1)

    # --- já organizado? ---
    if _find_split_root(DATASET_DIR) == DATASET_DIR:
        print("[INFO] Dataset já organizado em dataset/")
        verify()
        return

    # --- localiza a fonte ---
    source: Path = args.source or autodetect_source()

    if source is None:
        print("\n[ERRO] Não foi possível localizar o dataset.")
        print("       Baixe o arquivo do Kaggle e informe o caminho:")
        print("       python prepare_dataset.py --source archive.zip")
        print("       Link: https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8")
        sys.exit(1)

    if not source.exists():
        print(f"\n[ERRO] Arquivo/pasta não encontrado: {source}")
        sys.exit(1)

    # --- extrai se for zip ---
    if source.suffix.lower() == ".zip":
        extract_dir = Path("dataset")
        source = extract_zip(source, extract_dir)

    # --- encontra raiz com splits ---
    split_root = _find_split_root(source)
    if split_root is None:
        print(f"\n[ERRO] Não encontrei train/, valid/ ou test/ dentro de {source}")
        print("       Verifique se extraiu o arquivo correto do Kaggle.")
        sys.exit(1)

    # --- reorganiza ---
    reorganize(split_root)

    # --- verifica resultado ---
    ok = verify()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
