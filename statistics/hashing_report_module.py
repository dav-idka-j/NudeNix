from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class HashingReportModule(BaseReportModule):
    def get_title(self) -> str: return "Image Hashing (Optimization)"

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        hashing_time = stats.sums.get('hashing', 0)
        hashing_calls = stats.counters.get('hashing_calls', 0)
        skip_count = stats.counters.get('skip_count', 0)
        frame_count = stats.counters.get('frame_count', 0)

        hashing_pct = (hashing_time / duration * 100) if duration > 0 else 0
        skip_rate = (skip_count / frame_count * 100) if frame_count > 0 else 0

        return [
            f"Total Time: {hashing_time:.2f}s ({hashing_pct:.1f}%)",
            f"Hashing Runs: {hashing_calls}",
            f"Detections Skipped: {skip_count} ({skip_rate:.1f}% skip rate)"
        ]
