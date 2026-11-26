#!/usr/bin/env python3
from pathlib import Path

from g1_atom_player.config import load_config
from g1_atom_player.server import serve


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    serve(load_config(root))
