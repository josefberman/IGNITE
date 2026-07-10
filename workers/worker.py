"""Background worker entry (thread-compatible; Redis optional later)."""

from __future__ import annotations

import time

from backend.app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"IGNITE worker ready. data_dir={settings.data_dir}")
    print("Jobs currently run in-process via FastAPI threads.")
    print("Redis URL:", settings.redis_url or "(unset — in-process mode)")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
