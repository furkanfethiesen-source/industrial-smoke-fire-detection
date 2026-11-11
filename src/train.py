"""Training, Validation, and ONNX Export Pipeline for Smoke and Fire Detection.

Designed for rapid local verification on mini_dfire, with support for automatic
validation reporting and ONNX edge conversion. Main training with large epochs is
intended for Google Colab/cloud GPU environments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainPipeline")


def resolve_device(requested: str) -> str:
    """Selects best available hardware accelerator."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    data_yaml: str = "data/mini_dfire/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 3,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = "auto",
    project: str = "runs/detect",
    name: str = "smoke_fire_run",
    export_onnx: bool = True,
    benchmark_onnx: bool = True,
) -> Dict[str, Any]:
    """Runs YOLO training pipeline with validation and edge ONNX export."""
    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_path}")

    selected_device = resolve_device(device)
    logger.info("Initializing training pipeline...")
    logger.info("Dataset: %s", data_path)
    logger.info("Base model: %s", base_model)
    logger.info("Epochs: %d (local lightweight verification)", epochs)
    logger.info("Batch size: %d | Image size: %d | Device: %s", batch_size, img_size, selected_device)

    model = YOLO(base_model)

    start_train_time = time.perf_counter()
    train_results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=selected_device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    total_train_sec = time.perf_counter() - start_train_time
    logger.info("Training completed in %.2f seconds.", total_train_sec)

    # 1. Extract validation metrics and output directory
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(project) / name
    best_pt_path = save_dir / "weights" / "best.pt"
    if not best_pt_path.exists():
        best_pt_path = save_dir / "weights" / "last.pt"

    val_metrics: Dict[str, Any] = {}
    if hasattr(train_results, "results_dict"):
        for k, v in train_results.results_dict.items():
            val_metrics[k] = float(v) if isinstance(v, (int, float)) else str(v)

    logger.info("Validation Summary:")
    for k, v in val_metrics.items():
        logger.info("  %s: %s", k, v)

    onnx_path = None
    onnx_metrics: Dict[str, Any] = {}

    # 2. Export to ONNX if requested (FOCUS.md #3: Edge Inference)
    if export_onnx and best_pt_path.exists():
        logger.info("Exporting best weights to ONNX format...")
        best_model = YOLO(str(best_pt_path))
        exported_str = best_model.export(format="onnx", imgsz=img_size)
        onnx_path = Path(exported_str)
        logger.info("ONNX model saved at: %s", onnx_path)

        if benchmark_onnx:
            onnx_metrics = benchmark_inference(
                pt_path=best_pt_path,
                onnx_path=onnx_path,
                img_size=img_size,
                device=selected_device,
            )

    summary = {
        "dataset": str(data_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": selected_device,
        "training_time_seconds": round(total_train_sec, 2),
        "best_weights": str(best_pt_path),
        "onnx_model": str(onnx_path) if onnx_path else None,
        "metrics": val_metrics,
        "benchmark": onnx_metrics,
    }

    summary_file = save_dir / "pipeline_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Report saved to %s", summary_file)
    return summary


def benchmark_inference(
    pt_path: Path,
    onnx_path: Path,
    img_size: int = 640,
    device: str = "cpu",
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Compares inference latency fairly between PyTorch (.pt) and ONNX Runtime (.onnx).

    Uses identical end-to-end preprocessing (letterbox), inference on the same compute
    device, and NMS postprocessing via SmokeFireDetector for both backends.
    """
    import numpy as np
    import cv2
    import glob
    from src.detector import SmokeFireDetector

    logger.info("Running Fair Edge Benchmark: PyTorch vs ONNX Runtime on device: %s...", device)

    # Try to load a real sample image from validation/test data for realistic benchmark
    real_images = glob.glob("data/mini_dfire/images/test/*.jpg") + glob.glob("data/mini_dfire/images/val/*.jpg") + glob.glob("data/**/*.jpg", recursive=True)
    if real_images:
        loaded = cv2.imread(real_images[0])
        sample_input = loaded if loaded is not None else np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    else:
        sample_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 1. PyTorch Benchmark via SmokeFireDetector
    pt_detector = SmokeFireDetector(model_path=pt_path, img_size=img_size, device=device)
    for _ in range(num_warmup):
        _ = pt_detector.detect(sample_input)

    pt_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = pt_detector.detect(sample_input)
        pt_times.append(time.perf_counter() - t0)

    avg_pt_ms = (sum(pt_times) / len(pt_times)) * 1000.0
    pt_fps = 1000.0 / avg_pt_ms if avg_pt_ms > 0 else 0.0

    # 2. ONNX Runtime Benchmark via SmokeFireDetector on the same device
    onnx_detector = SmokeFireDetector(model_path=onnx_path, img_size=img_size, device=device)
    for _ in range(num_warmup):
        _ = onnx_detector.detect(sample_input)

    onnx_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = onnx_detector.detect(sample_input)
        onnx_times.append(time.perf_counter() - t0)

    avg_onnx_ms = (sum(onnx_times) / len(onnx_times)) * 1000.0
    onnx_providers = onnx_detector.ort_session.get_providers() if onnx_detector.ort_session else []

    logger.info("Benchmark Results (Device: %s):", device)
    logger.info("  PyTorch (.pt)   -> Avg: %.2f ms | FPS: %.1f", avg_pt_ms, pt_fps)
    logger.info("  ONNX (.onnx)    -> Avg: %.2f ms | FPS: %.1f | Providers: %s", avg_onnx_ms, onnx_fps, onnx_providers)
    if device == "mps":
        logger.info(
            "  [Hardware Note] PyTorch is executing on Apple Silicon GPU via Metal (MPS), "
            "while ONNX Runtime is executing on %s.",
            onnx_providers,
        )

    return {
        "device": device,
        "pytorch": {"avg_latency_ms": round(avg_pt_ms, 2), "fps": round(pt_fps, 1)},
        "onnx": {
            "avg_latency_ms": round(avg_onnx_ms, 2),
            "fps": round(onnx_fps, 1),
            "providers": onnx_providers,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Smoke & Fire Training and Export Pipeline")
    parser.add_argument("--data", type=str, default="data/mini_dfire/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model architecture or weights")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3 for quick verification)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip ONNX vs PyTorch benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_model(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            export_onnx=not args.no_export,
            benchmark_onnx=not args.no_benchmark,
        )
    except Exception as exc:
        logger.error("Training pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)
