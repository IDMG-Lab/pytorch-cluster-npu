from pathlib import Path
import sys
TEST=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(TEST/'precision/cases'),str(TEST/'precision/scripts')]
