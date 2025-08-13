"""Action module for defining callable actions with descriptions."""

from typing import Callable


class Action:
    """
    A class that represents an action with a description and an associated function.

    Attributes:
        description (str): A string describing what the action does.
        function (Callable): An arbitrary callable that implements the action.
    """

    def __init__(self, description: str, function: Callable):
        """
        Initialize an Action with a description and function.

        Args:
            description (str): A string describing what the action does.
            function (Callable): An arbitrary callable that implements the action.
        """
        self.description = description
        self.function = function

    def __call__(self, *args, **kwargs):
        """
        Make the Action instance callable by delegating to the function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of calling the function with the provided arguments.
        """
        return self.function(*args, **kwargs)

    def __str__(self) -> str:
        """
        Return a string representation of the Action.

        Returns:
            str: A string showing the description and function name.
        """
        func_name = getattr(self.function, '__name__', str(self.function))
        return f"Action(description='{self.description}', function={func_name})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Action.

        Returns:
            str: A detailed string representation.
        """
        return self.__str__()
