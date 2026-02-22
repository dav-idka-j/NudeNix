from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class OutputReportModule(BaseReportModule):
    def get_title(self) -> str: return "Output (Writer)"

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        output_time = stats.sums.get('output', 0)
        frame_count = stats.counters.get('frame_count', 0)
        output_pct = (output_time / duration * 100) if duration > 0 else 0
        output_avg = (output_time / frame_count * 1000) if frame_count > 0 else 0
        return [
            f"Total Time: {output_time:.2f}s ({output_pct:.1f}%)",
            f"Avg Time/Frame: {output_avg:.2f}ms"
        ]
