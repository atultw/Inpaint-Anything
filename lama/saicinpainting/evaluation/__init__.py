import logging


def make_evaluator(kind='default', **kwargs):
    """
    Dummy evaluator for prediction-only mode.
    This should never be called when predict_only=True.
    """
    logging.info(f'Make evaluator {kind}')
    raise NotImplementedError("Evaluator not available in minimal version")
