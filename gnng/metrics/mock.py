from torchmetrics import Metric


class MockMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, *args, **kwargs) -> None:
        return

    def compute(self):
        return
