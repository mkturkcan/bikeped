"""Locate, launch and stop a CARLA server process.

A CARLA release ships a launcher script (``CarlaUE4.sh`` in the Unreal
Engine 4 releases, ``CarlaUnreal.sh`` in the Unreal Engine 5 release) that
starts the simulator binary. This module finds an install, builds the launch
command for either engine, runs it in its own process group, waits until a
client can read the server version and stops the whole group again.

``carla`` is only needed by :meth:`ServerProcess.wait_ready`, which asks the
server for its version through a client.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import IO, Iterator, List, Optional, Sequence, Tuple, Union

from .engine import PROFILES, EngineProfile, carla_module

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

#: Values accepted by the UE4 ``-quality-level=`` flag.
QUALITY_LEVELS: Tuple[str, ...] = ("Epic", "Low")


def _profile(engine: str) -> EngineProfile:
    try:
        return PROFILES[engine]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown engine {engine!r}, expected one of: {known}") from None


def _quality(quality: str) -> str:
    for level in QUALITY_LEVELS:
        if level.lower() == str(quality).lower():
            return level
    raise ValueError(f"unknown quality {quality!r}, expected one of: {', '.join(QUALITY_LEVELS)}")


def launcher_path(engine: str, root: PathLike) -> Path:
    """Path of the launcher script inside the install at ``root``.

    The UE4 archives keep the launcher at the top of the install and the UE5
    archive keeps it under ``Linux/``, so both places are checked. When the
    script is in neither, the top-level path is returned so that callers can
    name it in their error message.
    """
    profile = _profile(engine)
    base = Path(root).expanduser()
    candidates = (base / profile.launcher, base / "Linux" / profile.launcher)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def find_install(engine: str, root: Optional[PathLike] = None) -> Path:
    """Return the root folder of a CARLA install for ``engine``.

    An explicit ``root`` is the only place searched when it is given. Without
    it, the engine's environment variable (``CARLA_UE4_ROOT`` or
    ``CARLA_UE5_ROOT``) is tried first and then the profile's list of common
    install paths.

    Raises:
        FileNotFoundError: no candidate contains the engine's launcher script.
    """
    profile = _profile(engine)
    if root is not None:
        base = Path(root).expanduser()
        if launcher_path(engine, base).is_file():
            return base
        raise FileNotFoundError(
            f"{profile.launcher} not found in {base} or {base / 'Linux'}. Point --root or "
            f"{profile.root_env} at the folder of the extracted CARLA release."
        )

    candidates: List[str] = []
    env_root = os.environ.get(profile.root_env)
    if env_root:
        candidates.append(env_root)
    candidates.extend(profile.search_paths)

    tried: List[str] = []
    for candidate in candidates:
        base = Path(os.path.expandvars(candidate)).expanduser()
        if launcher_path(engine, base).is_file():
            return base
        tried.append(str(base))
    looked = ", ".join(tried) if tried else "no candidate paths"
    raise FileNotFoundError(
        f"No CARLA {engine} install found (looked for {profile.launcher} in: {looked}). "
        f"Set {profile.root_env} or pass --root with the folder of the extracted CARLA release."
    )


def build_command(
    engine: str,
    root: Optional[PathLike] = None,
    port: int = 2000,
    quality: str = "Epic",
    windowed: bool = False,
    extra_args: Sequence[str] = (),
) -> List[str]:
    """Build the argument list that launches a server.

    The server renders off screen unless ``windowed`` is set. ``port`` is the
    RPC port; the server also uses the two ports after it for streaming.
    ``quality`` is passed as ``-quality-level=`` on UE4 and ignored on UE5,
    whose launcher has no such flag. With ``root`` left out the install is
    looked up with :func:`find_install`.
    """
    profile = _profile(engine)
    level = _quality(quality)
    base = find_install(engine) if root is None else root
    command = [str(launcher_path(engine, base))]
    for flag in profile.launch_flags:
        if windowed and flag.lower() == "-renderoffscreen":
            continue
        command.append(flag)
    if windowed:
        command.append("-windowed")
    command.append(f"-carla-rpc-port={int(port)}")
    if profile.supports_quality_flag:
        command.append(f"-quality-level={level}")
    command.extend(extra_args)
    return command


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ServerProcess:
    """One CARLA server started from a local install.

    The launcher script starts the simulator binary as a child process, so the
    server gets its own session and process group and :meth:`stop` signals the
    whole group. A terminal interrupt therefore reaches only the Python
    process, which is expected to call :meth:`stop` on its way out
    (:func:`managed_server` does).

    Args:
        engine: ``"ue4"`` or ``"ue5"``.
        root: install folder; looked up with :func:`find_install` when omitted.
        port: RPC port of the server.
        quality: UE4 quality level, ``"Epic"`` or ``"Low"``.
        windowed: open a window instead of rendering off screen.
        host: address used to check that the server answers.
        log_path: file that receives the server's output; a file in the system
            temporary folder by default.
        extra_args: further launcher arguments, appended as they are.
    """

    def __init__(
        self,
        engine: str,
        root: Optional[PathLike] = None,
        port: int = 2000,
        quality: str = "Epic",
        windowed: bool = False,
        host: str = "127.0.0.1",
        log_path: Optional[PathLike] = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.engine = engine
        self.profile = _profile(engine)
        self.root = root
        self.port = int(port)
        self.quality = _quality(quality)
        self.windowed = windowed
        self.host = host
        self.extra_args = tuple(extra_args)
        default_log = Path(tempfile.gettempdir()) / f"carla_{engine}_{self.port}.log"
        self.log_path = Path(log_path) if log_path is not None else default_log
        self.command: List[str] = []
        self.version: Optional[str] = None
        self._process: Optional["subprocess.Popen[bytes]"] = None
        self._pgid: Optional[int] = None
        self._log: Optional[IO[bytes]] = None

    @property
    def pid(self) -> Optional[int]:
        """Process id of the launcher, or ``None`` when no server is running."""
        return self._process.pid if self._process is not None else None

    @property
    def running(self) -> bool:
        """Whether any process of the server's group is still alive."""
        if self._process is None:
            return False
        if self._process.poll() is None:
            return True
        return self._group_alive()

    def start(self) -> "ServerProcess":
        """Launch the server and return immediately.

        Raises:
            RuntimeError: a server was already started from this object, or
                something already listens on the RPC port.
            FileNotFoundError: no install with the engine's launcher was found.
        """
        if self._process is not None:
            raise RuntimeError("this server was already started")
        if _port_open(self.host, self.port):
            raise RuntimeError(
                f"port {self.port} on {self.host} is already in use; stop the server that "
                "listens there or choose another --port"
            )
        root = find_install(self.engine, self.root)
        self.command = build_command(
            self.engine, root, self.port, self.quality, self.windowed, self.extra_args
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "wb")
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=str(Path(self.command[0]).parent),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            log.close()
            raise
        self._log = log
        # start_new_session makes the launcher the leader of a new process group.
        self._pgid = self._process.pid
        logger.info("started CARLA %s server (pid %d): %s", self.engine, self._process.pid,
                    " ".join(self.command))
        return self

    def wait_ready(self, timeout: float = 180.0, poll_interval: float = 1.0) -> str:
        """Block until a client can read the server version, and return it.

        Raises:
            RuntimeError: the server was not started or exited while starting up.
            TimeoutError: the server did not answer within ``timeout`` seconds.
        """
        if self._process is None:
            raise RuntimeError("the server has not been started")
        carla = carla_module()
        deadline = time.monotonic() + timeout
        last_error: Optional[Exception] = None
        while True:
            if not self.running:
                raise RuntimeError(
                    f"the CARLA server exited with code {self._process.returncode} before it "
                    f"accepted connections; its output is in {self.log_path}\n{self.log_tail()}"
                )
            try:
                client = carla.Client(self.host, self.port)
                client.set_timeout(2.0)
                self.version = str(client.get_server_version())
                logger.info("CARLA server %s is ready on port %d", self.version, self.port)
                return self.version
            except (RuntimeError, OSError) as error:
                last_error = error
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"no answer from the CARLA server on {self.host}:{self.port} after "
                    f"{timeout:.0f} s (last error: {last_error}); its output is in {self.log_path}"
                )
            time.sleep(min(poll_interval, remaining))

    def stop(self, grace: float = 10.0) -> None:
        """Stop the server and every process it started.

        The process group receives ``SIGTERM`` and, if it is still alive after
        ``grace`` seconds, ``SIGKILL``. Calling this on a server that is not
        running does nothing.
        """
        if self._process is None:
            return
        try:
            if self.running:
                self._signal_group(signal.SIGTERM)
                if not self._wait_exit(grace):
                    logger.warning("CARLA server ignored SIGTERM for %.0f s, killing it", grace)
                    self._signal_group(signal.SIGKILL)
                    if not self._wait_exit(grace):
                        logger.error("CARLA server process group %s is still alive", self._pgid)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
            self._process = None
            self._pgid = None

    def log_tail(self, lines: int = 20) -> str:
        """Last ``lines`` lines of the server's output, empty if unreadable."""
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _group_alive(self) -> bool:
        if self._pgid is None:
            return False
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _signal_group(self, signum: int) -> None:
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, signum)
        except ProcessLookupError:
            pass

    def _wait_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.running:
                return True
            time.sleep(0.2)
        return not self.running


@contextlib.contextmanager
def managed_server(
    engine: str,
    root: Optional[PathLike] = None,
    port: int = 2000,
    quality: str = "Epic",
    windowed: bool = False,
    timeout: float = 180.0,
    host: str = "127.0.0.1",
    log_path: Optional[PathLike] = None,
) -> Iterator[ServerProcess]:
    """Run a CARLA server for the duration of a ``with`` block.

    The server is started, the block is entered once a client can reach it,
    and the server is stopped when the block exits for any reason, including
    an exception or a keyboard interrupt.
    """
    server = ServerProcess(
        engine, root=root, port=port, quality=quality, windowed=windowed, host=host,
        log_path=log_path,
    )
    try:
        server.start()
        server.wait_ready(timeout=timeout)
        yield server
    finally:
        server.stop()
