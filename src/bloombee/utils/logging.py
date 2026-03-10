import logging
import os

from hivemind.utils import logging as hm_logging


def initialize_logs():
    """Initialize BloomBee logging tweaks. This function is called when you import the `bloombee` module."""

    # Env var BLOOMBEE_LOGGING=False prohibits BloomBee do anything with logs
    if os.getenv("BLOOMBEE_LOGGING", "True").lower() in ("false", "0"):
        return

    hm_logging.use_hivemind_log_handler("in_root_logger")

    # BLOOMBEE_DEBUG=1 enables dprint() output AND bloombee logger.debug() lines.
    # Only bloombee loggers go to DEBUG; hivemind stays at INFO to avoid Go p2p
    # daemon debug noise ([p2pd], [rtrefresh/...], [go-libp2p-kad-dht/...]).
    if os.getenv("BLOOMBEE_DEBUG", "0").lower() in ("1", "true", "yes", "on"):
        logging.getLogger("bloombee").setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers regardless of debug mode
    for noisy in ("urllib3", "requests", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var BLOOMBEE_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("BLOOMBEE_ASYNCIO_LOGLEVEL", "FATAL" if hm_logging.loglevel != "DEBUG" else "DEBUG")
    hm_logging.get_logger("asyncio").setLevel(asyncio_loglevel)
