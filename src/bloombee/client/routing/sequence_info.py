import dataclasses
import threading
import time
import weakref
from typing import Iterable, List, Optional, Tuple

from bloombee.data_structures import ModuleUID, RemoteModuleInfo, RemoteSpanInfo, ServerState
from bloombee.utils.hivemind_compat import get_logger
from bloombee.utils.dht import compute_spans

logger = get_logger(__name__)

# Legacy pickles do not contain the shared-view group. Keep weak references to
# recently unpickled views so views of the same block records can rejoin it.
_unpickled_views = []
_unpickled_views_lock = threading.Lock()


@dataclasses.dataclass
class _SharedSequenceViews:
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)
    views: list = dataclasses.field(default_factory=list, repr=False)

    def __getstate__(self):
        # Locks and weak references are process-local; rebuild them on load.
        return {}

    def __setstate__(self, state):
        self.lock = threading.Lock()
        self.views = []


@dataclasses.dataclass
class RemoteSequenceInfo:
    """
    A dataclass that stores general information about which servers hold any given layer;
    - updated by RemoteSequenceManager in a background thread
    - accessed by routing strategies in .on_update
    :note: this class should *not* be modified by RoutingStrategy.on_update to avoid interference between strategies;
     Any metadata specific to one routing strategy, it should be stored inside that strategy. Any information that
     is used by most routing strategies should be moved from said strategies to this class.
    """

    block_uids: Tuple[ModuleUID, ...]
    block_infos: Tuple[RemoteModuleInfo, ...]  # note: the contents of RemoteModuleInfo can and will be updated
    spans_by_priority: List[RemoteSpanInfo]
    spans_containing_block: Tuple[List[RemoteSpanInfo], ...]
    last_updated_time: Optional[float]

    def __post_init__(self):
        # Keep this outside dataclass fields so asdict() retains its original shape.
        self._shared_views = _SharedSequenceViews()
        self._shared_views.views.append(weakref.ref(self))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_shared_views" not in state:
            self._shared_views = _SharedSequenceViews()

        with _unpickled_views_lock:
            live_references = []
            shared_block_ids = {id(info) for info in self.block_infos}
            matching_groups = []
            for reference in _unpickled_views:
                view = reference()
                if view is not None:
                    live_references.append(reference)
                    if any(id(info) in shared_block_ids for info in view.block_infos):
                        if not any(group is view._shared_views for group in matching_groups):
                            matching_groups.append(view._shared_views)

            target_group = matching_groups[0] if matching_groups else self._shared_views
            if self._shared_views is not target_group:
                matching_groups.append(self._shared_views)
            for group in matching_groups:
                if group is target_group:
                    continue
                for reference in group.views:
                    view = reference()
                    if view is not None:
                        view._shared_views = target_group
                        target_group.views.append(reference)
                group.views = []

            self._shared_views = target_group
            target_group.views.append(weakref.ref(self))
            _unpickled_views[:] = live_references + [weakref.ref(self)]

    @property
    def lock_changes(self):
        return self._shared_views.lock

    @classmethod
    def make_empty(cls, block_uids: Iterable[ModuleUID]) -> "RemoteSequenceInfo":
        block_uids = tuple(block_uids)
        empty_block_infos = tuple(RemoteModuleInfo(uid, {}) for uid in block_uids)
        empty_spans = tuple([] for _ in range(len(block_uids)))
        return cls(block_uids, empty_block_infos, [], empty_spans, last_updated_time=None)

    def __getitem__(self, ix: slice):
        assert isinstance(ix, slice)
        with self.lock_changes:
            block_uids, block_infos = self.block_uids[ix], self.block_infos[ix]
            spans_by_priority, spans_containing_block = self._sort_spans(block_infos)
            view = RemoteSequenceInfo(
                block_uids, block_infos, spans_by_priority, spans_containing_block, self.last_updated_time
            )
            view._shared_views = self._shared_views
            self._shared_views.views.append(weakref.ref(view))
            return view

    def __len__(self):
        return len(self.block_uids)

    def update_(self, new_block_infos: List[RemoteModuleInfo]):
        assert len(new_block_infos) == len(self.block_uids)
        for block_index, (uid, info) in enumerate(zip(self.block_uids, new_block_infos)):
            assert uid == info.uid, f"The DHT entry for {uid} actually points to {info.uid}"
            self.block_infos[block_index].servers = info.servers

        self.rebuild_spans_()
        self.last_updated_time = time.perf_counter()

    def rebuild_spans_(self):
        """Keep every view of the shared block information consistent."""
        live_views = []
        for reference in self._shared_views.views:
            view = reference()
            if view is not None:
                view.spans_by_priority, view.spans_containing_block = self._sort_spans(view.block_infos)
                live_views.append(reference)
        self._shared_views.views = live_views

    def remove_peer_(self, peer_id):
        """Evict a failed peer from all views that share these block records."""
        seen_blocks = set()
        for reference in self._shared_views.views:
            view = reference()
            if view is not None:
                for info in view.block_infos:
                    if id(info) not in seen_blocks:
                        info.servers.pop(peer_id, None)
                        seen_blocks.add(id(info))
        self.rebuild_spans_()

    @staticmethod
    def _sort_spans(block_infos: List[RemoteModuleInfo]):
        spans_by_priority = list(compute_spans(block_infos, min_state=ServerState.ONLINE).values())
        spans_by_priority.sort(key=lambda span: span.length, reverse=True)

        spans_containing_block = tuple([] for _ in range(len(block_infos)))
        for span in spans_by_priority:
            for block_index in range(span.start, span.end):
                spans_containing_block[block_index].append(span)

        return spans_by_priority, spans_containing_block
