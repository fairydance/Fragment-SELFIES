"""Exception classes for Fragment-SELFIES."""


class FragmentSelfiesError(Exception):
    """Base exception for Fragment-SELFIES."""


class FragmentSelfiesEncodeError(FragmentSelfiesError):
    """Raised when a molecule cannot be encoded."""


class FragmentSelfiesDecodeError(FragmentSelfiesError):
    """Raised when a Fragment-SELFIES string cannot be decoded."""
