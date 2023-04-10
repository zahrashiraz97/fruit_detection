from struct import unpack, calcsize
from io import RawIOBase
from typing import TypeVar, Type


T = TypeVar('T')
_FORMAT_MAPS = {}


def cstruct(fmt: str):
    def make_cstruct(cls: type):
        _FORMAT_MAPS[cls] = fmt, calcsize(fmt)
        return cls
    return make_cstruct


def parse_cstruct(struct_type: Type[T], stream: RawIOBase) -> T:
    fmt, size = _FORMAT_MAPS[struct_type]
    buffer = stream.read(size)
    return struct_type(*unpack(fmt, buffer))
