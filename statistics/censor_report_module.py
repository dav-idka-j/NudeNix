from typing import List
from .base_report_module import BaseReportModule
from .performance_statistics import PerformanceStatistics

class CensorReportModule(BaseReportModule):
    def get_title(self) -> str: return "Censors (Processing)"

    def render(self, stats: PerformanceStatistics) -> List[str]:
        duration = stats.get_total_duration()
        censor_total_time = stats.sums.get('censors_total', 0)
        frames_with_censorship = stats.counters.get('frames_with_censorship', 0)
        censor_pct = (censor_total_time / duration * 100) if duration > 0 else 0

        lines = [
            f"Total Time: {censor_total_time:.2f}s ({censor_pct:.1f}%)",
            f"Frames with Censorship: {frames_with_censorship}"
        ]

        censor_calls_by_type = stats.categorized_counters.get('censor_calls_by_type', {})
        if censor_calls_by_type:
            lines.append("Breakdown by Censor Type:")
            for c_type, c_calls in sorted(censor_calls_by_type.items(), key=lambda x: x[1], reverse=True):
                c_time = stats.sums.get(f"censor_{c_type}_time", 0)
                avg = (c_time / c_calls * 1000) if c_calls > 0 else 0
                lines.append(f"  * {c_type}: {c_time:.2f}s ({c_calls} calls, avg {avg:.2f}ms/call)")
        return lines
