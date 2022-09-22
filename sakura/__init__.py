import os
import sys
from .functional import *
from gnutools import fs
__version__ = "0.1.1"
cfg = fs.load_config(f"{fs.parent(__file__)}/config.yml")
ascii_txt = """\
   _____           _                               __  __   _      
  / ____|         | |                             |  \/  | | |     
 | (___     __ _  | | __  _   _   _ __    __ _    | \  / | | |     
  \___ \   / _` | | |/ / | | | | | '__|  / _` |   | |\/| | | |     
  ____) | | (_| | |   <  | |_| | | |    | (_| |   | |  | | | |____ 
 |_____/   \__,_| |_|\_\  \__,_| |_|     \__,_|   |_|  |_| |______|
"""


def main():
    print(ascii_txt)
    os.system(
        f"mpirun --mca btl_base_warn_component_unused 0 -np 2 python -W ignore {' '.join(sys.argv[1:])}")
