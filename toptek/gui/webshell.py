"""Helpers to serve the React web console inside the desktop shell."""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import webbrowser
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from core import utils

__all__ = ["WebFrontendHandle", "find_web_build", "launch_web_frontend"]


@dataclass
class WebFrontendHandle:
    """Wrapper for a background HTTP server exposing the built assets."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    url: str

    def stop(self) -> None:
        """Stop the background server if it is still running."""

        if getattr(self.server, "__toptek_closed", False):  # type: ignore[attr-defined]
            return
        setattr(self.server, "__toptek_closed", True)  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            self.server.shutdown()
        with contextlib.suppress(Exception):
            self.server.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def open(self) -> None:
        """Open the served frontend in the default web browser."""

        webbrowser.open_new(self.url)


def find_web_build(paths: utils.AppPaths, relative: str = "toptek/ui/web/dist") -> Optional[Path]:
    """Return the packaged React build directory when available."""

    candidate = paths.root / relative
    index_file = candidate / "index.html"
    return candidate if index_file.exists() else None


def _pick_port(preferred: Optional[int] = None) -> int:
    if preferred:
        return preferred
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_web_frontend(
    paths: utils.AppPaths,
    *,
    port: Optional[int] = None,
    auto_open: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Optional[WebFrontendHandle]:
    """Serve built React assets from ``toptek/ui/web/dist``."""

    build_dir = find_web_build(paths)
    if build_dir is None:
        if logger:
            logger.info("Web frontend build missing. Run `npm run build` in toptek/ui/web first.")
        return None

    listen_port = _pick_port(port)
    handler = partial(SimpleHTTPRequestHandler, directory=str(build_dir))

    try:
        server = ThreadingHTTPServer(("127.0.0.1", listen_port), handler)
    except OSError as exc:
        if logger:
            logger.error("Failed to bind web frontend server on port %s: %s", listen_port, exc)
        return None

    thread = threading.Thread(target=server.serve_forever, name="ToptekWebFrontend", daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{listen_port}"
    if logger:
        logger.info("Serving React console from %s at %s", build_dir, url)
    if auto_open:
        webbrowser.open_new(url)
    return WebFrontendHandle(server=server, thread=thread, url=url)
