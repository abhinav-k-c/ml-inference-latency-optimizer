class ModelRouter:
    def __init__(self, latency_monitor, sla_ms=50):
        self.latency_monitor = latency_monitor
        self.sla_ms = sla_ms
        self.last_model = "large"

    def choose_model(self):
        if self.latency_monitor.sla_violated():
            self.last_model = "small"
            return "small"
        self.last_model = "large"
        return "large"
