import sys
from .functional import *
__version__ = "0.1.0"
import os
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
    os.system(f"mpirun -n 2 python {' '.join(sys.argv[1:])}")
