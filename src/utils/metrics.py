import time
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger("MetricsCollector")


@dataclass
class RoundMetrics:
    """Metrics for a single federated learning round."""

    round_num: int
    loss: float = 0.0
    accuracy: float = 0.0
    num_clients: int = 0
    wall_time: float = 0.0


class MetricsCollector:
    """Collects and stores metrics across federated learning rounds."""

    def __init__(self) -> None:
        self.rounds: list[RoundMetrics] = []
        self._round_start: float = 0.0

    def start_round(self) -> None:
        """Mark the start of a round for wall-time measurement."""
        self._round_start = time.monotonic()

    def record_round(
        self,
        round_num: int,
        loss: float = 0.0,
        accuracy: float = 0.0,
        num_clients: int = 0,
    ) -> RoundMetrics:
        """Record metrics for a completed round."""
        wall_time = time.monotonic() - self._round_start if self._round_start else 0.0
        metrics = RoundMetrics(
            round_num=round_num,
            loss=loss,
            accuracy=accuracy,
            num_clients=num_clients,
            wall_time=wall_time,
        )
        self.rounds.append(metrics)
        logger.info(
            f"Round {round_num}: loss={loss:.4f}, accuracy={accuracy:.4f}, "
            f"clients={num_clients}, time={wall_time:.1f}s"
        )
        return metrics

    def get_latest(self) -> RoundMetrics | None:
        """Return the most recent round metrics."""
        return self.rounds[-1] if self.rounds else None

    def summary(self) -> dict[str, float]:
        """Return a summary of all recorded metrics."""
        if not self.rounds:
            return {}
        accuracies = [r.accuracy for r in self.rounds]
        losses = [r.loss for r in self.rounds]
        return {
            "num_rounds": len(self.rounds),
            "best_accuracy": max(accuracies),
            "final_accuracy": accuracies[-1],
            "best_loss": min(losses),
            "final_loss": losses[-1],
            "total_wall_time": sum(r.wall_time for r in self.rounds),
        }
