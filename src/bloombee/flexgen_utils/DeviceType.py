

from enum import Enum, auto

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()
    DISK = auto()
    MIXED = auto()
    COMPRESSED = auto()

    @staticmethod
    def convert(name):
        if name == "cpu":
            return DeviceType.CPU
        elif name == "cuda":
            return DeviceType.CUDA
        elif name == "disk":
            return DeviceType.DISK
        elif name == "mixed":
            return DeviceType.MIXED
        elif name == "compressed":
            return DeviceType.COMPRESSED
        else:
            raise ValueError(f"Invalid name: {name}")