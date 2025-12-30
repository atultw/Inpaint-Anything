class NoopVisualizer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        pass
