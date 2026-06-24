"""Command line interface for Fragment-SELFIES."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

from .brics import (
    build_vocabulary_from_smiles,
    iter_smiles_file,
    verify_brics_fragment_invariance,
)
from .codec import FRAGMENT_STYLES, FragmentSelfiesCodec
from .vocabulary import FragmentVocabulary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _count_smiles_records(path: Path, *, max_molecules: int | None) -> int:
    count = 0
    for _smiles in iter_smiles_file(path):
        count += 1
        if max_molecules is not None and count >= max_molecules:
            break
    return count


def _cmd_build_vocab(args: argparse.Namespace) -> None:
    progress = sys.stderr.isatty() if args.progress is None else args.progress
    progress_bar = None
    progress_callback = None
    if progress:
        progress_bar = tqdm(
            total=_count_smiles_records(args.input, max_molecules=args.max_molecules),
            desc="Building vocabulary",
            unit="mol",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} molecules [{elapsed}<{remaining}, {rate_fmt}]",
            file=sys.stderr,
        )
        progress_callback = progress_bar.update
    try:
        vocab, stats = build_vocabulary_from_smiles(
            iter_smiles_file(args.input),
            canonical_input=args.canonical,
            max_molecules=args.max_molecules,
            min_count=args.min_count,
            max_fragments=args.max_fragments,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
            progress_callback=progress_callback,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    vocab.save_jsonl(args.output)
    print(json.dumps({"vocab_size": len(vocab), **asdict(stats)}, indent=2, sort_keys=True))


def _cmd_verify_brics(args: argparse.Namespace) -> None:
    result = verify_brics_fragment_invariance(
        iter_smiles_file(args.input),
        sample_size=args.sample_size,
        randomizations=args.randomizations,
        seed=args.seed,
    )
    print(json.dumps(asdict(result) | {"is_consistent": result.is_consistent}, indent=2, sort_keys=True))


def _load_codec(args: argparse.Namespace) -> FragmentSelfiesCodec:
    if getattr(args, "vocab", None) is not None:
        return FragmentSelfiesCodec(FragmentVocabulary.load_jsonl(args.vocab))
    return FragmentSelfiesCodec()


def _cmd_encode(args: argparse.Namespace) -> None:
    codec = _load_codec(args)
    print(
        codec.encode(
            args.smiles,
            fallback_selfies=not args.no_fallback_selfies,
            canonical=args.canonical,
            randomized=args.randomized,
            seed=args.seed,
            implicit_probability=args.implicit_probability,
            max_implicit_cuts=args.max_implicit_cuts,
            fragment_style=args.fragment_style,
        )
    )


def _cmd_encode_fragment(args: argparse.Namespace) -> None:
    codec = _load_codec(args)
    print(
        codec.encode_fragment(
            args.smiles,
            canonical=args.canonical,
            randomized=args.randomized,
            seed=args.seed,
            include_fragment_token=not args.body_only,
        )
    )


def _cmd_decode(args: argparse.Namespace) -> None:
    codec = _load_codec(args)
    mol = codec.decode(
        args.fragment_selfies,
        strict=args.strict,
        repair_missing_return_attachment=args.repair_missing_return_attachment,
    )
    print(
        Chem.MolToSmiles(
            mol,
            canonical=args.canonical,
            doRandom=args.randomized,
            isomericSmiles=False,
        )
    )


def _cmd_decode_fragment(args: argparse.Namespace) -> None:
    codec = _load_codec(args)
    mol = codec.decode_fragment(args.fragment_selfies)
    print(
        Chem.MolToSmiles(
            mol,
            canonical=args.canonical,
            doRandom=args.randomized,
            isomericSmiles=False,
        )
    )


def _cmd_reserialize(args: argparse.Namespace) -> None:
    codec = _load_codec(args)
    print(
        codec.reserialize(
            args.fragment_selfies,
            fallback_selfies=not args.no_fallback_selfies,
            canonical=args.canonical,
            randomized=args.randomized,
            seed=args.seed,
            implicit_probability=args.implicit_probability,
            max_implicit_cuts=args.max_implicit_cuts,
            fragment_style=args.fragment_style,
            strict=args.strict,
            repair_missing_return_attachment=args.repair_missing_return_attachment,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fragment-SELFIES utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-vocab", help="Build a BRICS fragment vocabulary from an SMI file")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--min-count", type=int, default=1)
    build.add_argument("--max-fragments", type=int)
    build.add_argument("--max-molecules", type=int)
    build.add_argument("--workers", type=_positive_int, default=os.cpu_count() or 1, help="Number of worker processes")
    build.add_argument("--chunk-size", type=_positive_int, default=200, help="Molecules per worker task")
    build.add_argument("--canonical", action="store_true", help="Canonicalize input SMILES before fragmentation")
    progress = build.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show build progress on stderr; use conda run --live-stream for live updates",
    )
    progress.add_argument("--no-progress", dest="progress", action="store_false", help="Do not show build progress")
    build.set_defaults(progress=None)
    build.set_defaults(func=_cmd_build_vocab)

    verify = sub.add_parser("verify-brics", help="Verify BRICS fragment invariance under randomized SMILES")
    verify.add_argument("--input", required=True, type=Path)
    verify.add_argument("--sample-size", type=int, default=1000)
    verify.add_argument("--randomizations", type=int, default=5)
    verify.add_argument("--seed", type=int, default=0)
    verify.set_defaults(func=_cmd_verify_brics)

    encode = sub.add_parser("encode", help="Encode one SMILES string")
    encode.add_argument("--vocab", type=Path, help=argparse.SUPPRESS)
    encode.add_argument("--smiles", required=True)
    encode_mode = encode.add_mutually_exclusive_group()
    encode_mode.add_argument("--canonical", action="store_true", help="Canonicalize input before encoding")
    encode_mode.add_argument("--randomized", action="store_true", help="Randomize atom and traversal order")
    encode.add_argument("--seed", type=int, help="Seed randomized encoding for reproducibility")
    encode.add_argument(
        "--implicit-probability",
        type=float,
        default=0.15,
        help="Probability that randomized auto style emits an implicit adjacent-root edge",
    )
    encode.add_argument(
        "--max-implicit-cuts",
        type=int,
        default=1,
        help="Maximum implicit BRICS edges to cut per connected component",
    )
    encode.add_argument(
        "--fragment-style",
        "--style",
        choices=sorted(FRAGMENT_STYLES),
        default="auto",
        help="Output style: auto, explicit edges, or implicit adjacent-root anchors",
    )
    encode.add_argument("--no-fallback-selfies", action="store_true")
    encode.set_defaults(func=_cmd_encode)

    encode_fragment = sub.add_parser(
        "encode-fragment",
        help="Encode one connected SMILES fragment",
    )
    encode_fragment.add_argument("--vocab", type=Path, help=argparse.SUPPRESS)
    encode_fragment.add_argument("--smiles", required=True)
    encode_fragment_mode = encode_fragment.add_mutually_exclusive_group()
    encode_fragment_mode.add_argument(
        "--canonical",
        action="store_true",
        help="Canonicalize input before encoding",
    )
    encode_fragment_mode.add_argument(
        "--randomized",
        action="store_true",
        help="Randomize atom order",
    )
    encode_fragment.add_argument("--seed", type=int, help="Seed randomized encoding for reproducibility")
    encode_fragment.add_argument(
        "--body-only",
        action="store_true",
        help="Emit only the fragment body tokens, without the leading [Frag] token",
    )
    encode_fragment.set_defaults(func=_cmd_encode_fragment)

    decode = sub.add_parser("decode", help="Decode one Fragment-SELFIES string")
    decode.add_argument("--vocab", type=Path, help=argparse.SUPPRESS)
    decode.add_argument("--fragment-selfies", required=True)
    decode.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Require strict Fragment-SELFIES decoding. Default is non-strict recovery mode.",
    )
    decode.add_argument(
        "--repair-missing-return-attachment",
        action="store_true",
        help="Repair linker outputs by adding one safe return attachment at the farthest suitable atom.",
    )
    decode_mode = decode.add_mutually_exclusive_group()
    decode_mode.add_argument("--canonical", action="store_true", help="Canonicalize output SMILES")
    decode_mode.add_argument("--randomized", action="store_true", help="Randomize output SMILES")
    decode.set_defaults(func=_cmd_decode)

    decode_fragment = sub.add_parser(
        "decode-fragment",
        help="Decode one Fragment-SELFIES fragment and preserve dummy atoms",
    )
    decode_fragment.add_argument("--vocab", type=Path, help=argparse.SUPPRESS)
    decode_fragment.add_argument("--fragment-selfies", required=True)
    decode_fragment_mode = decode_fragment.add_mutually_exclusive_group()
    decode_fragment_mode.add_argument(
        "--canonical",
        action="store_true",
        help="Canonicalize output fragment SMILES",
    )
    decode_fragment_mode.add_argument(
        "--randomized",
        action="store_true",
        help="Randomize output fragment SMILES",
    )
    decode_fragment.set_defaults(func=_cmd_decode_fragment)

    reserialize = sub.add_parser(
        "reserialize",
        aliases=["convert"],
        help="Decode and re-encode Fragment-SELFIES in a requested style",
    )
    reserialize.add_argument("--vocab", type=Path, help=argparse.SUPPRESS)
    reserialize.add_argument("--fragment-selfies", required=True)
    reserialize.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Require strict Fragment-SELFIES decoding before re-encoding. Default is non-strict recovery mode.",
    )
    reserialize.add_argument(
        "--repair-missing-return-attachment",
        action="store_true",
        help="Repair linker outputs by adding one safe return attachment before re-encoding.",
    )
    reserialize_mode = reserialize.add_mutually_exclusive_group()
    reserialize_mode.add_argument("--canonical", action="store_true", help="Canonicalize before re-encoding")
    reserialize_mode.add_argument("--randomized", action="store_true", help="Randomize before re-encoding")
    reserialize.add_argument("--seed", type=int, help="Seed randomized re-encoding")
    reserialize.add_argument(
        "--fragment-style",
        "--style",
        choices=sorted(FRAGMENT_STYLES),
        default="explicit",
        help="Output style: auto, explicit edges, or implicit adjacent-root anchors",
    )
    reserialize.add_argument(
        "--implicit-probability",
        type=float,
        default=0.15,
        help="Probability used when --style auto and --randomized are selected",
    )
    reserialize.add_argument(
        "--max-implicit-cuts",
        type=int,
        default=1,
        help="Maximum implicit BRICS edges to cut per connected component",
    )
    reserialize.add_argument("--no-fallback-selfies", action="store_true")
    reserialize.set_defaults(func=_cmd_reserialize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
