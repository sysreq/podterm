from __future__ import annotations

import urllib.error
import urllib.request


class EventdClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url
        self.token = token

    def get(self, path: str, timeout: float) -> tuple[int, object, bytes]:
        """GET base_url+path. Returns (status, headers, body); status 0 on connection error."""
        # Cloudflare (in front of proxy.runpod.net) 403s the default Python-urllib UA.
        req = urllib.request.Request(
            self.base_url + path,
            headers={"Authorization": f"Bearer {self.token}", "User-Agent": "podterm/2.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, b""
        except Exception:
            return 0, None, b""
