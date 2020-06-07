from collections import defaultdict
from ignite import metrics


class LossAverager(metrics.Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.count = 0
        self.summation = defaultdict(int)

    def update(self, output):
        losses, batch_size = output

        self.count += batch_size
        for key, value in output[0].items():
            self.summation[key] += batch_size * value

    def compute(self):
        results = {k: v / self.count for k, v in self.summation.items()}
        return results
