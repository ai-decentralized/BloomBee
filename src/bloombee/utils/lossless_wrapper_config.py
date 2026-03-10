"""
Central switches for BloomBee lossless transport wrapper.

Edit values in this file before starting client/server processes.
"""

# 0 = disable lossless wrapper, 1 = enable
ENABLE_LOSSLESS_WRAPPER = 0

# "zstd" (recommended), "zlib", "zipnn", "none"
# Note: current ZipNN integration is only lossless on dtypes that pass the runtime self-check.
LOSSLESS_ALGO = "zstd"

# Compression level:
# - zstd: typically 1..22 (3 is a good low-latency default)
# - zlib: -1..9
LOSSLESS_LEVEL = 3

# "plain" = compress the original serialized buffer as one stream
# "byte_split" = also try splitting float16/float32 high-byte lanes into a second zstd stream
# Ignored when LOSSLESS_ALGO == "zipnn"
LOSSLESS_LAYOUT = "byte_split"

# source:channel:tensor_name selectors; "*" is allowed per field
# Used only when LOSSLESS_LAYOUT == "byte_split"
LOSSLESS_LAYOUT_TARGETS = "*:*:hidden_states"

# Only compress if serialized tensor buffer size >= LOSSLESS_MIN_BYTES
LOSSLESS_MIN_BYTES = 49152

# Require at least this many bytes saved, otherwise keep original buffer
LOSSLESS_MIN_GAIN_BYTES = 2048

# 1 = environment variables can override this file
# 0 = this file has full control
ALLOW_ENV_OVERRIDE = 1

# Optional research-only side profiling.
# When 1, non-ZipNN runs may also evaluate ZipNN on the same raw buffer for comparison.
# Default off to keep zstd / byte_split benchmarks clean.
COMP_ZIPNN_PROFILE = 0
