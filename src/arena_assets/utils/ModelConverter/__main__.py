import os
import sys

from .converter import ModelConverter


def main(*args: str) -> None:
    if not args:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input_file> [<output_file>] ...")
        return
    with ModelConverter() as converter:
        converter.load(args[0])
        converter.rectify()
        if len(args) >= 2:
            converter.save(args[1])


main(*sys.argv[1:])
