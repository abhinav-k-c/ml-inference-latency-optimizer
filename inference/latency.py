from collections import deque
import statistics

class LatencyMonitor:
    def __init__(self, window_size=50, sla_ms=50):
        self.latencies = deque(maxlen=window_size)
        self.sla_ms = sla_ms
        self.total_requests = 0
        self.sla_violations = 0

    def record(self, latency_ms):
        self.latencies.append(latency_ms)
        self.total_requests += 1
        if self.avg_latency() > self.sla_ms:
            self.sla_violations += 1

    def avg_latency(self):
        return statistics.mean(self.latencies) if self.latencies else 0.0

    def p95_latency(self):
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[int(0.95 * len(sorted_lat)) - 1]

    def sla_violated(self):
        return self.avg_latency() > self.sla_ms