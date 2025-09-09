"""
I love enums, so we need some enum helpers
"""

from enum import Enum


class BaseEnum(Enum):
    """
    Some base enum functionality to allow checking for value and name
    presence.
    """

    @classmethod
    def has_value(cls, value):
        """
        Check if value is part of the enum
        """
        return value in cls._value2member_map_

    @classmethod
    def has_name(cls, name):
        """
        Check if name is part of the enum
        """
        return name in cls.__members__

    @classmethod
    def list(cls) -> list:
        """
        Get a list of all enum values
        """
        return list(map(lambda c: c.value, cls))
