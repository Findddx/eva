"""Opt-in owner of one request's statistical profile; never GPU placement.

The caller supplies ProfileIdentity, scopes the observer around real target
work, and calls finish only after a successful request. All storage is bounded
aggregate metadata: no prompt text, token IDs, embeddings, weights or KV.
"""
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from ds41.personal_profile import PersonalExpertProfile, PHASES
from ds41.route_observer import TargetRouteObserver


MAX_METADATA_BYTES = 16 * 1024**2
IDENTITY_SCOPE = "runtime_metadata_fingerprint_not_whole_weight_authentication"


class ProfileBusyError(RuntimeError):
    pass


@contextmanager
def exclusive_profile_lock(path):
    """Linux/POSIX nonblocking flock; retain the lock inode after release.

    Never unlink a lock file: another process could otherwise lock a new inode
    while an old holder still owns the original. Process exit closes the fd.
    """
    if os.name != "posix":
        raise RuntimeError("personal profile sessions require POSIX fcntl.flock; unsupported platform")
    import fcntl
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("profile lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            raise ProfileBusyError(f"personal profile already has an active request: {path}") from error
        acquired = True
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def canonical_metadata_fingerprint(metadata):
    """Return a labelled metadata digest, explicitly not a weight checksum."""
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                     allow_nan=False, ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("identity metadata exceeds bounded size")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "scope": IDENTITY_SCOPE}


def _sync_directory(path):
    if os.name != "posix":
        return False
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _temporary(path):
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".session-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _write_payload(path, payload):
    if not isinstance(payload, bytes) or len(payload) > MAX_METADATA_BYTES:
        raise ValueError("candidate must be bounded metadata bytes")
    with path.open("wb") as stream:
        view = memoryview(payload)
        while view:
            written = stream.write(view)
            if written is None or written <= 0:
                raise OSError("short candidate write")
            view = view[written:]
        stream.flush()
        os.fsync(stream.fileno())


class PersonalProfileSession:
    """One lock, one request, one explicit finish attempt; close never saves.

    ``lock_factory(path)`` is an optional context-manager factory for CPU tests.
    The default is the real nonblocking POSIX flock above. An enter/load error
    releases it. finish retains it until close/__exit__.
    """

    def __init__(self, path, identity, *, update_every=50, lock_factory=None):
        if type(update_every) is not int or update_every < 1:
            raise ValueError("update_every must be a positive integer")
        raw = Path(path).absolute()
        raw.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = raw.parent.resolve(strict=True) / raw.name
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.identity = identity
        self.update_every = update_every
        self._lock_factory = exclusive_profile_lock if lock_factory is None else lock_factory
        self._lock = None
        self._entered = self._closed = self._finish_attempted = False
        self.profile = self.request = self.observer = None
        self.last_report = None

    def enter(self):
        if self._entered or self._closed:
            raise ValueError("profile session is single-use")
        self._lock = self._lock_factory(self.lock_path)
        try:
            self._lock.__enter__()
        except BaseException:
            self._lock = None
            self._closed = True
            raise
        try:
            if os.path.lexists(self.path):
                if self.path.is_symlink() or not self.path.is_file():
                    raise ValueError("working profile must be a regular non-symlink file")
                self.profile = PersonalExpertProfile.load_state(self.path, self.identity)
                if self.profile.update_every != self.update_every:
                    raise ValueError("stored profile interval differs from configured update_every")
            else:
                self.profile = PersonalExpertProfile(self.identity, update_every=self.update_every)
            self._previous_count = self.profile.completed_requests
            self.request = self.profile.begin_request()
            self.observer = TargetRouteObserver(self.request, enabled=True)
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    __enter__ = enter

    @property
    def latest_candidate_path(self):
        """Latest completed boundary, never the current partial window.

        This is a deterministic path, not a file-existence fallback search.
        The caller must load/verify the file and its identity before any use.
        """
        if self.profile is None:
            raise ValueError("enter the profile session before selecting a candidate")
        boundary = self.profile.completed_requests // self.profile.update_every * self.profile.update_every
        if not boundary:
            return None
        return self.path.with_name(f"{self.path.stem}.candidate-{boundary:06d}.json")

    def _report(self, status, candidate=None):
        latest = self.latest_candidate_path
        return dict(kind="personal_profile_session", status=status,
                    identity_sha256=self.identity.sha256, identity_scope=IDENTITY_SCOPE,
                    state_path=str(self.path), lock_path=str(self.lock_path),
                    candidate_path=None if candidate is None else str(candidate),
                    latest_candidate_path=None if latest is None else str(latest),
                    previous_completed_requests=self._previous_count,
                    completed_requests=self.profile.completed_requests,
                    update_every=self.profile.update_every,
                    max_routed_slots=self.observer.max_routed_slots,
                    calls={phase: int(self.request.calls[i]) for i, phase in enumerate(PHASES)},
                    route_uses={phase: int(self.request.route_uses[i]) for i, phase in enumerate(PHASES)},
                    call_union_presence_totals={phase: int(self.request.presence[i].sum())
                                                for i, phase in enumerate(PHASES)},
                    saves_prompt_or_token_ids=False, GPU_placement_executed=False,
                    placement_scope="session statistics owner only; caller startup application is separate")

    def finish(self, *, processed_prefill_tokens, committed_decode_tokens, valid_pre_eos=True):
        if not self._entered or self._closed or self._finish_attempted:
            raise ValueError("finish requires an active session and may be attempted only once")
        if type(valid_pre_eos) is not bool:
            raise ValueError("valid_pre_eos must be a bool")
        if any(type(value) is not int or value < 0 for value in
               (processed_prefill_tokens, committed_decode_tokens)):
            raise ValueError("token exposure counts must be nonnegative integers")
        self._finish_attempted = True
        if not valid_pre_eos:
            self.last_report = self._report("discarded")
            self.last_report.update(state_saved=False, candidate_published=False,
                                    processed_prefill_tokens=processed_prefill_tokens,
                                    committed_decode_tokens=committed_decode_tokens)
            return self.last_report
        snapshot = self.profile.complete_request(
            self.request, processed_prefill_tokens=processed_prefill_tokens,
            committed_decode_tokens=committed_decode_tokens, valid_pre_eos=True)
        candidate = (self.path.with_name(f"{self.path.stem}.candidate-{snapshot.trained_through_request:06d}.json")
                     if snapshot is not None else None)
        report = self._report("saving", candidate)
        report.update(state_saved=False, candidate_published=False,
                      processed_prefill_tokens=processed_prefill_tokens,
                      committed_decode_tokens=committed_decode_tokens,
                      publication="candidate first, then working; not a two-file atomic transaction")
        working_temp = candidate_temp = None
        try:
            if candidate is not None and os.path.lexists(candidate):
                raise FileExistsError(f"refusing to overwrite existing boundary candidate: {candidate}")
            # Stage all data before changing either public path. save_state is
            # itself bounded, validates the working schema and fsyncs its file.
            working_temp = _temporary(self.path)
            self.profile.save_state(working_temp)
            if candidate is not None:
                candidate_temp = _temporary(candidate)
                _write_payload(candidate_temp, snapshot.payload)
                # link is atomic no-replace. Both files live in the same dir;
                # unsupported filesystems fail closed, never fall back to copy.
                os.link(candidate_temp, candidate)
                report["candidate_published"] = True
                candidate_temp.unlink()
                candidate_temp = None
                _sync_directory(self.path.parent)
            os.replace(working_temp, self.path)
            working_temp = None
            report["state_saved"] = True
            report["directory_synced"] = _sync_directory(self.path.parent)
            report["status"] = "saved"
            self.last_report = report
            return report
        except BaseException as error:
            retained, cleanup_errors = [], []
            for temporary in (working_temp, candidate_temp):
                if temporary is not None and temporary.exists():
                    if report["candidate_published"] or report["state_saved"]:
                        retained.append(str(temporary))
                    else:
                        try:
                            temporary.unlink()
                        except OSError as cleanup_error:
                            retained.append(str(temporary))
                            cleanup_errors.append(str(cleanup_error))
            report.update(status="storage_failed", retained_temporary_files=retained,
                          cleanup_errors=cleanup_errors)
            self.last_report = report
            error.personal_profile_status = report
            raise

    def close(self):
        """Abort/release only. No implicit completion or filesystem save."""
        self._closed = True
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.__exit__(None, None, None)

    def __exit__(self, *_exception):
        self.close()
        return False
