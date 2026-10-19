"""Compatibility entry for the framework's evaluation client."""
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import client
    if Path(client.__file__).resolve() != root / "client.py":
        raise RuntimeError("Evaluation client must come from this framework checkout")
    return client.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
