from collections import deque
import statistics

class LatencyMonitor:
    def __init__(self, window_size=50, sla_ms=50):
        self.latencies = deque(maxlen=window_size)
        self.sla_ms = sla_ms

    def record(self, latency_ms):
        self.latencies.append(latency_ms)

    def avg_latency(self):
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    def p95_latency(self):
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        index = int(0.95 * len(sorted_lat)) - 1
        return sorted_lat[max(index, 0)]

    def sla_violated(self):
        return self.avg_latency() > self.sla_ms
__all__ = ["LatencyMonitor"]
