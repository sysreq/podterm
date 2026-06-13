"""Entry point for podterm: uv run podterm"""

import logging

import uvicorn

from podterm.config import load_dotenv


class _PodPollFilter(logging.Filter):
    """Suppress the constant GET /api/pods access-log spam."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return '"GET /api/pods ' not in msg


def main() -> None:
    load_dotenv()  # local secrets (e.g. RUNPOD_CONSOLE_CLIENT_COOKIE) before app start
    logging.getLogger("uvicorn.access").addFilter(_PodPollFilter())
    uvicorn.run("podterm.server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
