from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class DetectionReportModule(BaseReportModule):
    def get_title(self) -> str: return "Detection (NudeNet)"

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        detect_time = stats.sums.get('detection', 0)
        detection_runs = stats.counters.get('detection_runs', 0)

        detect_pct = (detect_time / duration * 100) if duration > 0 else 0
        detect_avg = (detect_time / detection_runs * 1000) if detection_runs > 0 else 0

        lines = [
            f"Total Time: {detect_time:.2f}s ({detect_pct:.1f}%)",
            f"Inference Runs: {detection_runs}",
            f"Avg Time/Inference: {detect_avg:.2f}ms"
        ]

        label_counts = stats.categorized_counters.get('label_frequency', {})
        total_detections = sum(label_counts.values())
        if total_detections > 0:
            lines.append(f"Detections Found: {total_detections}")
            lines.append("Label Frequency:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  * {label}: {count}")
        return lines
