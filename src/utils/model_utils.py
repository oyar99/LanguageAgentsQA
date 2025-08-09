"""Utilities for models."""


def supports_temperature_param(model: str) -> bool:
    """
    Checks if the model supports temperature parameter.

    Args:
        model (str): the model identifier

    Returns:
        bool: True if the model supports temperature parameter, False otherwise
    """
    # Models that support temperature parameter
    models_with_temp = [
        'gpt-4o-mini',
        'gpt-4o-mini-2',
        'gpt-4o-mini-batch',
        'Qwen/Qwen2.5-14B-Instruct',
    ]

    return model in models_with_temp


def supports_batch(model: str) -> bool:
    """
    Checks if the model supports batch processing.

    Args:
        model (str): the model identifier

    Returns:
        bool: True if the model supports batch processing, False otherwise
    """
    # Models that support batch processing
    models_with_batch = [
        'gpt-4o-mini-batch',
        'gpt-4o-mini',
        'o3-mini',
    ]

    return model in models_with_batch
