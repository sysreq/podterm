"""Entry point for podterm: uv run podterm"""

import uvicorn


def main() -> None:
    uvicorn.run("podterm.server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
