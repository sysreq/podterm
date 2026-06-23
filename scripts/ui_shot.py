"""Full-page screenshot of the GPT Caddy UI for visual-diff checkpoints.

Usage: .venv/bin/python scripts/ui_shot.py OUT.png [--width 1456] [--height 1086] [--url URL]

Requires the chromium headless-shell system libs extracted to ~/.local/pw-libs
(see LD_LIBRARY_PATH below — installed without root on this machine).
Prints any browser console errors so checkpoints double as console-error checks.
"""

import argparse
import os
import sys
from pathlib import Path

LIB_DIR = Path.home() / ".local/pw-libs/extracted/usr/lib/x86_64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = (
    f"{LIB_DIR}:{LIB_DIR / 'nss'}:" + os.environ.get("LD_LIBRARY_PATH", "")
)

from playwright.sync_api import sync_playwright  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1456)
    ap.add_argument("--height", type=int, default=1086)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--wait", type=float, default=2.0, help="settle seconds after load")
    args = ap.parse_args()

    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(args.wait * 1000)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=args.out, full_page=True)
        browser.close()

    print(f"saved {args.out} ({args.width}x{args.height})")
    if errors:
        print(f"CONSOLE ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        return 1
    print("no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
