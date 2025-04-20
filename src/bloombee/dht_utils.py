import warnings

warnings.warn(
    "bloombee.dht_utils has been moved to bloombee.utils.dht. This alias will be removed in BloomBee 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from bloombee.utils.dht import *
