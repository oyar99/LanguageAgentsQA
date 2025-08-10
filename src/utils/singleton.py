"""A metaclass for singleton classes."""


from abc import ABCMeta


class Singleton(ABCMeta):
    """
    A metaclass to ensure a given class is only instantiated once. 

    Args:
        type (type): the metaclass to be used for the class

    Returns:
        type (type): the singleton instance of the class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
