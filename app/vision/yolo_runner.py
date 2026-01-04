from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    cls_name: str
    conf: float


class YoloRunner:
    """
    Wraps Ultralytics YOLO to output a consistent list of Detection objects.
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame_bgr: np.ndarray, imgsz: int, conf_thres: float) -> List[Detection]:
        results = self.model.predict(source=frame_bgr, imgsz=imgsz, conf=conf_thres, verbose=False)
        detections: List[Detection] = []

        for r in results:
            names = r.names
            boxes = r.boxes
            if boxes is None:
                continue

            for (xyxy, cls, conf) in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                cls_id = int(cls)
                cls_name = str(names.get(cls_id, cls_id))
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        cls_id=cls_id, cls_name=cls_name, conf=float(conf),
                    )
                )

        return detections
