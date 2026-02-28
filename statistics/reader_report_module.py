from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class ReaderReportModule(BaseReportModule):
    def get_title(self) -> str: return "Reader (Input)"

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        reader_time = stats.sums.get('reader', 0)
        frame_count = stats.counters.get('frame_count', 0)
        reader_pct = (reader_time / duration * 100) if duration > 0 else 0
        reader_avg = (reader_time / frame_count * 1000) if frame_count > 0 else 0
        return [
            f"Total Time: {reader_time:.2f}s ({reader_pct:.1f}%)",
            f"Avg Time/Frame: {reader_avg:.2f}ms"
        ]
