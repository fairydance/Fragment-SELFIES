"""Fragment-SELFIES public API."""

from .brics import (
    BRICSInvarianceResult,
    build_vocabulary_from_smiles,
    brics_fragment_smiles,
    verify_brics_fragment_invariance,
)
from .codec import (
    FRAGMENT_STYLE_AUTO,
    FRAGMENT_STYLE_EXPLICIT,
    FRAGMENT_STYLE_IMPLICIT,
    FragmentSelfiesCodec,
    decode,
    decode_fragment,
    encode,
    encode_fragment,
    reserialize,
)
from .exceptions import (
    FragmentSelfiesDecodeError,
    FragmentSelfiesEncodeError,
    FragmentSelfiesError,
)
from .vocabulary import FragmentEntry, FragmentVocabulary

__all__ = [
    "BRICSInvarianceResult",
    "FragmentEntry",
    "FragmentSelfiesCodec",
    "FragmentSelfiesDecodeError",
    "FragmentSelfiesEncodeError",
    "FragmentSelfiesError",
    "FRAGMENT_STYLE_AUTO",
    "FRAGMENT_STYLE_EXPLICIT",
    "FRAGMENT_STYLE_IMPLICIT",
    "FragmentVocabulary",
    "brics_fragment_smiles",
    "build_vocabulary_from_smiles",
    "decode",
    "decode_fragment",
    "encode",
    "encode_fragment",
    "reserialize",
    "verify_brics_fragment_invariance",
]

__version__ = "1.0.0"
