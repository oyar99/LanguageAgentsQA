"""
Utility functions for argparse
"""

import argparse


class KeyValue(argparse.Action):
    """
    Custom argparse action to parse key=value pairs into a dictionary.
    """
    # Constructor calling
    def __call__(self, _, namespace,
                 values, __=None):
        setattr(namespace, self.dest, {})

        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value
