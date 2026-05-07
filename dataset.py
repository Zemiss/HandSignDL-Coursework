from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition.preprocessing import *  # noqa: F401,F403,E402
