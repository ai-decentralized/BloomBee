"""Compatibility helpers for transformers APIs that moved/were removed in 5.x.

transformers 5.x removed ``transformers.utils.get_file_from_repo``.
The huggingface_hub ``hf_hub_download`` + ``try_to_load_from_cache`` pair
provides the equivalent semantics (return path or None, with local-only mode).
"""

from typing import Optional, Union

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError


def get_file_from_repo(
    path_or_repo_id: str,
    filename: str,
    *,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
    local_files_only: bool = False,
    **kwargs,
) -> Optional[str]:
    """Drop-in replacement for removed transformers.utils.get_file_from_repo.

    Returns the local filesystem path of a cached/downloaded file, or None
    when the file is absent. Matches the old contract used throughout the
    BloomBee codebase.
    """
    effective_token = token if token is not None else use_auth_token

    if local_files_only:
        cached = try_to_load_from_cache(
            repo_id=path_or_repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
        )
        if cached is None or cached == "_CACHED_NO_EXIST":
            return None
        return cached

    try:
        return hf_hub_download(
            repo_id=path_or_repo_id,
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
            token=effective_token,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None
