"""Create an Engram runtime configuration from a user-supplied split GGUF."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True,
                        help='The first file, ending in -00001-of-00007.gguf')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = args.index.expanduser().resolve(strict=True)
    match = re.fullmatch(r'(.*)-00001-of-00007\.gguf', str(index))
    if not match:
        parser.error('Expected the first piece of the complete seven-piece Engram GGUF')
    for number in range(1, 8):
        path = Path(f'{match[1]}-{number:05d}-of-00007.gguf')
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f'Missing or empty Engram piece: {path.name}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump({'format': 'ranma_mxfp8_gguf', 'index_path': str(index)}, handle, indent=2)
        handle.write('\n')
    print(f'Engram configuration written to {args.output}')


if __name__ == '__main__':
    main()
