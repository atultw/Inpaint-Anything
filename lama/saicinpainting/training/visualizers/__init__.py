import logging

from saicinpainting.training.visualizers.noop import NoopVisualizer


def make_visualizer(kind, **kwargs):
    logging.info(f'Make visualizer {kind}')

    if kind == 'noop':
        return NoopVisualizer()

    raise ValueError(f'Unknown visualizer kind {kind}')
