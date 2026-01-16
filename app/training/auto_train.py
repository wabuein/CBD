from __future__ import annotations

import os
import random
import shutil
from pathlib import Path


def ensure_fruit_model(weights_out: str) -> str:
    """
    Ensures a fruit-specific YOLO model exists at weights_out.

    If not found, it will:
      - download fruit subset from Open Images V7 (detections) via FiftyOne
      - export to COCO (single folder export with labels.json)
      - convert COCO -> YOLO labels
      - create YOLO-style train/val split folders and populate images
      - write fruits.yaml
      - train YOLO
      - save best weights to weights_out

    Returns weights_out.
    """
    weights_out = str(Path(weights_out))
    Path(os.path.dirname(weights_out)).mkdir(parents=True, exist_ok=True)

    if Path(weights_out).exists():
        print(f"[SETUP] Found fruit weights: {weights_out}")
        return weights_out

    print("[SETUP] Fruit weights not found. Starting auto dataset + training...")

    # --------- deps (import here so main app can run without training deps installed) ----------
    import fiftyone as fo
    import fiftyone.zoo as foz
    from ultralytics import YOLO
    from ultralytics.data.converter import convert_coco

    # --------- configure classes ----------
    classes = [
        "Apple",
        "Banana",
        "Orange",
        "Strawberry",
        "Pineapple",
        "Pear",
        "Coconut",
    ]

    # --------- step 1: download Open Images V7 subset ----------
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=12000,
        dataset_name="fruits_oi_train",
    )

    val = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=classes,
        max_samples=2000,
        dataset_name="fruits_oi_val",
    )

    # --------- step 2: export to COCO detection dataset ----------
    # NOTE: FiftyOne COCODetectionDataset export commonly produces:
    #   export_dir/data        (all images in one folder)
    #   export_dir/labels.json (COCO annotations)
    export_dir = Path("data/fruits_coco").resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] Exporting COCO to {export_dir} ...")
    ds.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )
    val.export(
        export_dir=str(export_dir),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )

    # --------- step 3: COCO -> YOLO conversion (labels) + create train/val images folders ----------
    coco_json = export_dir / "labels.json"
    if not coco_json.exists():
        raise RuntimeError(
            f"[SETUP] COCO annotations not found at {coco_json}\n"
            f"Expected FiftyOne export to create labels.json in {export_dir}"
        )

    yolo_root = Path("coco_converted2").resolve()
    yolo_root.mkdir(parents=True, exist_ok=True)

    print(f"[SETUP] Converting COCO->YOLO labels into {yolo_root} ...")
    # Important: convert_coco does NOT copy images; it generates label txt files.
    # We pass the COCO JSON directly (not an 'annotations' folder).
    convert_coco(
        str(coco_json),
        save_dir=str(yolo_root),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

    # Source images live here (single folder)
    all_images_dir = export_dir / "data"
    if not all_images_dir.exists():
        raise RuntimeError(
            f"[SETUP] Expected images folder not found: {all_images_dir}\n"
            f"Your export_dir contents: {list(export_dir.iterdir())}"
        )

    # Destination YOLO folder structure
    images_train = yolo_root / "images" / "train"
    images_val = yolo_root / "images" / "val"
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in all_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if len(imgs) < 50:
        raise RuntimeError(f"[SETUP] Too few images found in {all_images_dir}: {len(imgs)}")

    # Split images 90/10
    random.seed(42)
    random.shuffle(imgs)
    split_idx = int(0.9 * len(imgs))
    train_list = imgs[:split_idx]
    val_list = imgs[split_idx:]

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        # Hardlink is fast and doesn't duplicate data; fallback to copy if not supported
        try:
            os.link(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))

    print(f"[SETUP] Populating images: {len(train_list)} train, {len(val_list)} val ...")
    for p in train_list:
        _link_or_copy(p, images_train / p.name)
    for p in val_list:
        _link_or_copy(p, images_val / p.name)

    # Sanity: ensure labels exist (convert creates yolo_root/labels/**)
    labels_dir = yolo_root / "labels"
    if not labels_dir.exists():
        raise RuntimeError(f"[SETUP] No labels folder found after conversion: {labels_dir}")

    # --------- step 3.5: write fruits.yaml ----------
    yaml_path = Path("data/fruits.yaml").resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_text = (
        f"path: {yolo_root}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
    )
    for i, c in enumerate(classes):
        yaml_text += f"  {i}: {c}\n"

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[SETUP] Wrote dataset yaml: {yaml_path}")

    # --------- step 4: train YOLO ----------
    base = "models/yolo11n.pt"  # make sure this exists
    if not Path(base).exists():
        raise RuntimeError(
            f"[SETUP] Base weights not found: {base}\n"
            f"Put yolo11n.pt at models/yolo11n.pt or change 'base' path."
        )

    print("[SETUP] Training YOLO... this is a one-time setup step.")
    model = YOLO(base)

    results = model.train(
        data=str(yaml_path),
        imgsz=512,
        epochs=50,
        batch=16,
        device=0, #ensuring it runs on the device's gpu
        patience=10,
        workers=4,
    )

    # --------- step 5: copy best.pt to your models path ----------
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"[SETUP] Training finished but best.pt not found at: {best}")

    shutil.copyfile(str(best), weights_out)
    print(f"[SETUP] Saved fruit weights -> {weights_out}")

    return weights_out
