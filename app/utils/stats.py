from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, Tuple, List


class RunningStats:
    """Online mean/min/max (no need to store all samples)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def summary(self, unit: str = "") -> str:
        if self.n == 0:
            return "n=0"
        return f"avg={self.mean:.3f}{unit} min={self.min:.3f}{unit} max={self.max:.3f}{unit} n={self.n}"


@dataclass
class DetectionSummary:
    """
    Records detection summaries for a period.
    'Accuracy' requested by you is implemented as a practical proxy: mean confidence.
    True accuracy requires ground-truth labels (not available for live webcam frames).
    """
    det_conf: RunningStats = field(default_factory=RunningStats)
    det_count: int = 0

    class_counts: Counter = field(default_factory=Counter)
    color_counts: Counter = field(default_factory=Counter)
    class_color_counts: Counter = field(default_factory=Counter)  # (class, color) -> count

    class_conf_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    class_conf_n: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_detection(self, cls_name: str, color: str, conf: float) -> None:
        self.det_count += 1
        self.det_conf.add(conf)

        self.class_counts[cls_name] += 1
        self.color_counts[color] += 1
        self.class_color_counts[(cls_name, color)] += 1

        self.class_conf_sum[cls_name] += float(conf)
        self.class_conf_n[cls_name] += 1

    def class_mean_conf(self) -> Dict[str, float]:
        out = {}
        for k, s in self.class_conf_sum.items():
            n = self.class_conf_n.get(k, 0)
            out[k] = (s / n) if n else 0.0
        return out

    def top_classes(self, k: int = 5) -> List[Tuple[str, int]]:
        return self.class_counts.most_common(k)

    def top_colors(self, k: int = 5) -> List[Tuple[str, int]]:
        return self.color_counts.most_common(k)

    def top_class_color_pairs(self, k: int = 8) -> List[Tuple[Tuple[str, str], int]]:
        return self.class_color_counts.most_common(k)

    def reset(self) -> None:
        # Reset all counters/stats for a new trial
        self.det_conf = RunningStats()
        self.det_count = 0
        self.class_counts = Counter()
        self.color_counts = Counter()
        self.class_color_counts = Counter()
        self.class_conf_sum = defaultdict(float)
        self.class_conf_n = defaultdict(int)
