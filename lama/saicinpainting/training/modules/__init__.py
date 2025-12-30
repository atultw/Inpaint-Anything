import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator


def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')
    raise NotImplementedError("Discriminator not available in minimal version")
