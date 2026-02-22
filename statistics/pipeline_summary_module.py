from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class PipelineSummaryModule(BaseReportModule):
    def get_title(self) -> str: return "" # Summary doesn't get a numbered header

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        frame_count = stats.counters.get('frame_count', 0)
        fps = frame_count / duration if duration > 0 else 0
        return [
            f"Total Frames Processed: {frame_count}",
            f"Total Processing Time: {duration:.2f}s (~{fps:.2f} FPS)"
        ]
