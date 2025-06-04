#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_frames.py — extrait toutes les images d’une vidéo et les sauvegarde.

Usage :
    python extract_frames.py --input chemin/vers/video.mp4 --output dossier_frames --ext png

Arguments :
    --input   Chemin de la vidéo à traiter.
    --output  Dossier où seront écrites les images (créé s’il n’existe pas).
    --ext     Extension/format des images de sortie (jpg|png|bmp…).  [par défaut : jpg]
"""

import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, ext: str = "jpg") -> None:
    """Extrait chaque frame de la vidéo et la sauvegarde sous forme d’image."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Impossible de trouver la vidéo : {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ouvre la vidéo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Impossible d’ouvrir la vidéo avec OpenCV.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    digits = len(str(total)) if total > 0 else 6  # nombre de zéros pour le nommage

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        filename = output_dir / f"{idx:0{digits}d}.{ext}"
        # Pour jpg, on peut spécifier la qualité : cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(filename), frame)
        idx += 1

        # Affichage facultatif de progression
        if total:
            print(f"\rProgression : {idx}/{total} frames", end="")

    cap.release()
    print(f"\nTerminé ! {idx} images enregistrées dans : {output_dir.resolve()}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrait toutes les frames d’une vidéo.")
    parser.add_argument("--input", "-i", required=True, help="Vidéo source (mp4, avi, mov…).")
    parser.add_argument("--output", "-o", required=True, help="Dossier de sortie pour les images.")
    parser.add_argument("--ext", default="jpg", help="Extension/format pour les images (jpg, png, bmp…).")
    args = parser.parse_args()

    extract_frames(Path(args.input), Path(args.output), args.ext.lower())

if __name__ == "__main__":
    main()
