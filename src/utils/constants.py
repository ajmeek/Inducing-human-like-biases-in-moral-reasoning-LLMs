from dataclasses import dataclass

@dataclass(frozen=True)
class Sampling:
    LAST = 'last'
    AVG = 'avg'
    MIDDLE = 'middle'
    SENTENCES = 'sentences'
    METHODS = [ LAST, AVG, MIDDLE, SENTENCES ]
    ONE_POINT_METHODS = [ LAST, AVG, MIDDLE ]

@dataclass(frozen=True)
class FMRI:
    TR = 2
    REACT_TIME = 3 // TR

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