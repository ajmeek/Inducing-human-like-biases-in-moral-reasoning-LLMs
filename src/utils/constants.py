from dataclasses import dataclass
from enum import Flag, auto

class Sampling(Flag):
    LAST = auto()
    AVG = auto()
    MIDDLE = auto()
    SENTENCES = auto()
    ONE_POINT_METHODS = LAST | AVG | MIDDLE

@dataclass(frozen=True)
class FMRI:
    HEMODYNAMIC_LAG = 6
    TR = 2
    REACT_TIME = HEMODYNAMIC_LAG // TR

@dataclass(frozen=True)
class DS000212:
    @dataclass(frozen=True)
    class Periods:
        BACKGROUND = 6
        ACTION = 4
        OUTCOME = 4
        INTENT = 4
        JUDGMENT = 4
        ENDS = [
            BACKGROUND // FMRI.TR,
            ACTION // FMRI.TR,
            OUTCOME // FMRI.TR,
            INTENT // FMRI.TR,
            JUDGMENT // FMRI.TR
        ]