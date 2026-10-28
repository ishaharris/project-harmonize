"""Save credentials through hidden, interactive Terminal prompts."""

import sys
from getpass import getpass
from pathlib import Path

from dotenv import set_key


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Run this script directly in Terminal for hidden key entry.")
    path = Path(__file__).resolve().parents[1] / ".env"
    if path.is_symlink():
        raise SystemExit("Expected a regular project .env file, not a symbolic link.")
    print("Paste each key and press Enter. Keys stay hidden. Enter alone keeps the current value.")
    for name, label in (
        ("NVIDIA_API_KEY", "NVIDIA API key"),
        ("BIOHUB_API_TOKEN", "Biohub API token"),
    ):
        value = getpass(f"{label}: ").strip()
        if not value:
            print(f"{name}: unchanged.")
            continue
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise SystemExit("Key contains a control character; stopped without saving this key.")
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        set_key(str(path), name, value, quote_mode="always")
        path.chmod(0o600)
        print(f"{name}: saved.")
    print("Done. No predictions were submitted.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nStopped. Any keys already saved remain in .env.") from None
