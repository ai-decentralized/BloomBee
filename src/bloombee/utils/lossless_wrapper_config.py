"""
Central switches for BloomBee lossless transport wrapper.

Edit values in this file before starting client/server processes.
"""

# 0 = disable lossless wrapper, 1 = enable
ENABLE_LOSSLESS_WRAPPER = 0

# "zstd" (recommended), "zlib", "none"
LOSSLESS_ALGO = "zstd"

# Compression level:
# - zstd: typically 1..22 (3 is a good low-latency default)
# - zlib: -1..9
LOSSLESS_LEVEL = 3

# Only compress if serialized tensor buffer size >= LOSSLESS_MIN_BYTES
LOSSLESS_MIN_BYTES = 49152

# Require at least this many bytes saved, otherwise keep original buffer
LOSSLESS_MIN_GAIN_BYTES = 2048

# 1 = environment variables can override this file
# 0 = this file has full control
ALLOW_ENV_OVERRIDE = 1
