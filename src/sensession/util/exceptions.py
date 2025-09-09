"""
Collection of custom exceptions thrown by this library
"""


class NoDataError(Exception):
    """
    Exception thrown when no data was collected but apparent error was
    detected
    """


class InsufficientDataError(Exception):
    """
    Exception thrown when a prescribed data quota for captures was not met.
    """


class MultipleDataError(Exception):
    """
    Raised when a Tool reports more than one data group. Per instance, in campaign runs,
    we expect that not to happen. There, we perform the following sequence:

    Setup -> Run -> Reap

    In between, all states should be cleared, and so every tool must report one group.
    """


class SubprocessError(Exception):
    """
    Exception thrown when a subprocess returned an unexpected error code
    or a crash was detected
    """


class ApiUsageError(Exception):
    """
    Unfortunately, some APIs are a bit clunky and require specifis usages.
    If incompliancy to one of them was detected, this error is thrown.
    """


class MatlabError(Exception):
    """
    MATLAB running is quite finicky. We reraise any caught Matlab errors as this
    type so we don't need to import the optional matlab dependency to catch them
    """
