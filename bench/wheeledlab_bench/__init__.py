"""wheeledlab_bench: one measurement harness for roadmap 094's sides (scensim).

Stage timers around the env step and the RL runner's phases, GPU memory, and the learning
curve, written to one CSV schema (`record.COLUMNS`). It imports no WheeledLab code, so ours
(scensim's batched env) uses it unchanged.
"""
from .bench import Bench
from .record import COLUMNS, STAGES
from .timers import StageTimer

__all__ = ["Bench", "COLUMNS", "STAGES", "StageTimer"]
__version__ = "0.1.0"
