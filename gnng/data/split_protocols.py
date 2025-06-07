from enum import Enum


class SplitProtocol(Enum):
    PublicFixed = "public"
    RandSplit = "rand_split"
    RandSplitClass = "rand_split_class"
    PublicSemiFixed = "public_semi"
    SupervisedFixed = "public_supervised"
