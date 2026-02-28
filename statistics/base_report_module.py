from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .performance_statistics import PerformanceStatistics

class BaseReportModule(ABC):
    """
    Abstract Base Class for report sections.
    Components can implement this to define how their statistics are rendered.
    """
    @abstractmethod
    def get_title(self) -> str:
        """Returns the title of the report section."""
        pass

    @abstractmethod
    def render(self, stats: 'PerformanceStatistics') -> List[str]:
        """Returns a list of formatted lines for this section."""
        pass
