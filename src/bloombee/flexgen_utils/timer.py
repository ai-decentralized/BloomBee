"""Timer utilities."""

import time
from collections import defaultdict

class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = defaultdict(lambda: 0.0)
        self.starts = {}

    def start(self, name):
        """Start a timer."""
        self.starts[name] = time.time()

    def stop(self, name):
        """Stop a timer."""
        elapsed_time = time.time() - self.starts[name]
        self.timers[name] += elapsed_time

    def reset(self, name):
        """Reset a timer."""
        self.timers[name] = 0.0

    def reset_all(self):
        """Reset all timers."""
        self.timers.clear()
        self.starts.clear()

    def get(self, name):
        """Get a timer."""
        return self.timers[name]

    def log(self, name, normalizer=1.0):
        """Log a timer."""
        elapsed_time = self.timers[name]
        print(f"{name}: {elapsed_time/normalizer:.2f}")

    def log_all(self, normalizer=1.0):
        """Log all timers."""
        for name in self.timers:
            self.log(name, normalizer)

timers = Timers() 