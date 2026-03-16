import logging
import os

from hivemind.utils import logging as hm_logging

from bloombee.utils.debug_config import is_global_debug_enabled, is_log_channel_enabled


class _BloomBeeNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        if "[HANDLER_STEP_TIMING]" in message and not is_log_channel_enabled("handler_step_timing_logs"):
            return False

        if "[COMP_ZIPNN]" in message and not is_log_channel_enabled("zipnn_logs"):
            return False

        if "[KV_SOURCE_PROBE]" in message and not is_log_channel_enabled("kv_source_probe_logs"):
            return False

        if "[CROSS_GPU_TRANSFER_START]" in message and not is_log_channel_enabled("cross_gpu_transfer_logs"):
            return False

        if "[NETWORK_TX]" in message and not is_log_channel_enabled("client_inference_logs"):
            return False

        if "[CLIENT_SERVER_END]" in message and not is_log_channel_enabled("client_inference_logs"):
            return False

        if "[CLIENT_INFERENCE_END]" in message and not is_log_channel_enabled("client_inference_logs"):
            return False

        if "_ServerInferenceSession  step id " in message and not is_log_channel_enabled("client_inference_logs"):
            return False

        if "server inference session self._position:" in message and not is_log_channel_enabled("client_inference_logs"):
            return False

        if "[MBPIPE" in message:
            if "TimingSummary" in message or "[TIMING_SUMMARY]" in message:
                return True
            if not is_log_channel_enabled("microbatch_logs"):
                return False

        return True


def _install_bloombee_noise_filter() -> None:
    noise_filter = _BloomBeeNoiseFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(noise_filter)


def initialize_logs():
    """Initialize BloomBee logging tweaks. This function is called when you import the `bloombee` module."""

    # Env var BLOOMBEE_LOGGING=False prohibits BloomBee do anything with logs
    if os.getenv("BLOOMBEE_LOGGING", "True").lower() in ("false", "0"):
        return

    hm_logging.use_hivemind_log_handler("in_root_logger")
    _install_bloombee_noise_filter()

    # BLOOMBEE_DEBUG=1 enables dprint() output AND bloombee logger.debug() lines.
    # Only bloombee loggers go to DEBUG; hivemind stays at INFO to avoid Go p2p
    # daemon debug noise ([p2pd], [rtrefresh/...], [go-libp2p-kad-dht/...]).
    if is_global_debug_enabled():
        logging.getLogger("bloombee").setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers regardless of debug mode
    for noisy in ("urllib3", "requests", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var BLOOMBEE_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("BLOOMBEE_ASYNCIO_LOGLEVEL", "FATAL" if hm_logging.loglevel != "DEBUG" else "DEBUG")
    hm_logging.get_logger("asyncio").setLevel(asyncio_loglevel)
