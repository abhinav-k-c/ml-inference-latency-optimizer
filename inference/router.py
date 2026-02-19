class ModelRouter:
    def __init__(self, latency_monitor):
        self.latency_monitor = latency_monitor
        self.last_model = "large"
        self.switch_count = 0
        self.last_reason = "initial"

    def choose_model(self):
        if self.latency_monitor.sla_violated():
            if self.last_model != "small":
                self.switch_count += 1
            self.last_model = "small"
            self.last_reason = "SLA violated → switching to small model"
        else:
            if self.last_model != "large":
                self.switch_count += 1
            self.last_model = "large"
            self.last_reason = "SLA healthy → using large model"

        return self.last_model

    def get_debug_info(self):
        return {
            "last_reason": self.last_reason,
            "switch_count": self.switch_count,
        }
