import unittest
import time
from typing import List
from statistics import PerformanceStatistics, BaseReportModule

class MockReportModule(BaseReportModule):
    def __init__(self, title: str, lines: List[str]):
        self.title = title
        self.lines = lines
    def get_title(self) -> str: return self.title
    def render(self, stats: PerformanceStatistics) -> List[str]: return self.lines

class TestPerformanceStatistics(unittest.TestCase):
    def test_disabled_by_default(self):
        stats = PerformanceStatistics(enabled=False)
        self.assertFalse(stats.enabled)

        # Test that timer does nothing
        with stats.timer('test'):
            time.sleep(0.01)

        self.assertEqual(stats.sums['test'], 0.0)
        self.assertEqual(stats.counters['test_calls'], 0)

    def test_enabled_timer(self):
        stats = PerformanceStatistics(enabled=True)
        self.assertTrue(stats.enabled)

        with stats.timer('test'):
            time.sleep(0.01)

        self.assertGreater(stats.sums['test'], 0.0)
        self.assertEqual(stats.counters['test_calls'], 1)

    def test_module_registration_and_priority(self):
        stats = PerformanceStatistics(enabled=True)
        module1 = MockReportModule("Section B", ["Line 1"])
        module2 = MockReportModule("Section A", ["Line 2"])

        stats.register_module(module1, priority=20)
        stats.register_module(module2, priority=10)

        # Modules should be sorted by priority in print_report
        sorted_modules = sorted(stats._modules, key=lambda x: x[0])
        self.assertEqual(sorted_modules[0][1], module2)
        self.assertEqual(sorted_modules[1][1], module1)

    def test_print_report_with_modules(self):
        import io
        from contextlib import redirect_stdout

        stats = PerformanceStatistics(enabled=True)
        stats.increment_counter('frame_count', 1)
        stats.register_module(MockReportModule("Test Section", ["Value: 42"]))

        f = io.StringIO()
        with redirect_stdout(f):
            stats.print_report()
        output = f.getvalue()

        self.assertIn("NudeNix Performance & Analysis Report", output)
        self.assertIn("1. Test Section", output)
        self.assertIn("Value: 42", output)

    def test_generic_counter(self):
        stats = PerformanceStatistics(enabled=True)
        stats.increment_counter('skip_count', 2)
        self.assertEqual(stats.counters['skip_count'], 2)

    def test_record_sum(self):
        stats = PerformanceStatistics(enabled=True)
        stats.record_sum('test_sum', 0.5)
        stats.record_sum('test_sum', 0.3)
        self.assertEqual(stats.sums['test_sum'], 0.8)

    def test_record_categorized_counter(self):
        stats = PerformanceStatistics(enabled=True)
        stats.record_categorized_counter('label_frequency', 'FACE_FEMALE')
        stats.record_categorized_counter('label_frequency', 'FACE_FEMALE')
        stats.record_categorized_counter('label_frequency', 'FEMALE_BREAST_EXPOSED')

        self.assertEqual(stats.categorized_counters['label_frequency']['FACE_FEMALE'], 2)
        self.assertEqual(stats.categorized_counters['label_frequency']['FEMALE_BREAST_EXPOSED'], 1)

if __name__ == '__main__':
    unittest.main()
