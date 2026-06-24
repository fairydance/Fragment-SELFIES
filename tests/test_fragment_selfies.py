import json
import os
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from fragment_selfies import (
    FragmentSelfiesCodec,
    build_vocabulary_from_smiles,
    brics_fragment_smiles,
    decode_fragment,
    encode_fragment,
    verify_brics_fragment_invariance,
)
from fragment_selfies.cli import build_parser, main
from fragment_selfies.tokens import split_tokens


SAMPLE_SMI_ENV = "FRAGMENT_SELFIES_SAMPLE_SMI"


def same_molecule(left: str, right: str) -> bool:
    return Chem.MolToSmiles(Chem.MolFromSmiles(left), canonical=True, isomericSmiles=False) == Chem.MolToSmiles(
        Chem.MolFromSmiles(right), canonical=True, isomericSmiles=False
    )


def canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=False)


def root_fragment_count(fragment_selfies: str) -> int:
    return sum(1 for token in split_tokens(fragment_selfies) if token == "[Frag]")


def dummy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


def test_build_vocab_and_roundtrip_aspirin():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    vocab, stats = build_vocabulary_from_smiles([smiles])
    assert stats.molecules_used == 1
    assert len(vocab) > 1

    codec = FragmentSelfiesCodec()
    encoded = codec.encode(smiles, fallback_selfies=False)
    tokens = split_tokens(encoded)
    assert "[Frag]" in tokens
    assert "[Attach:" in encoded
    assert "[Dummy" in encoded
    assert "[Xe" not in encoded
    assert not any(token.startswith("[F:") for token in tokens)

    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)
    assert same_molecule(smiles, decoded)


def test_fragment_smiles_with_dummy_attachment_roundtrips_as_fragment():
    codec = FragmentSelfiesCodec()
    encoded = codec.encode_fragment("[*]C(=O)O", canonical=True)

    assert encoded.startswith("[Frag]")
    assert "[Dummy" in encoded
    assert "[Attach:" not in encoded

    decoded = codec.decode_fragment(encoded)
    decoded_smiles = Chem.MolToSmiles(decoded, canonical=True, isomericSmiles=False)

    assert decoded_smiles == "*C(=O)O"
    assert dummy_atom_count(decoded) == 1


def test_fragment_module_helpers_accept_body_only_fragment_selfies():
    body = encode_fragment("[*]N", canonical=True, include_fragment_token=False)
    decoded = decode_fragment(body)

    assert not body.startswith("[Frag]")
    assert Chem.MolToSmiles(decoded, canonical=True, isomericSmiles=False) == "*N"
    assert dummy_atom_count(decoded) == 1


def test_no_brics_cut_molecule_is_single_selfies_fragment():
    smiles = "c1ccccc1"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(smiles, fallback_selfies=False, randomized=False)
    tokens = split_tokens(encoded)
    assert tokens[0] == "[Frag]"
    assert tokens.count("[Frag]") == 1
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True)
    assert same_molecule(smiles, decoded)


def test_fragment_absent_from_vocab_still_uses_standard_selfies_tokens():
    codec = FragmentSelfiesCodec()
    encoded = codec.encode("c1ccccc1O")

    assert encoded.startswith("[Frag]")
    assert "[F:" not in encoded
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True)
    assert same_molecule("c1ccccc1O", decoded)


def test_brics_fragment_smiles_stable_for_randomized_smiles():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    baseline = Counter(brics_fragment_smiles(Chem.MolToSmiles(mol, canonical=True), canonical_input=False))
    for _ in range(10):
        randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        assert Counter(brics_fragment_smiles(randomized, canonical_input=False)) == baseline


def test_invariance_helper_reports_consistency():
    result = verify_brics_fragment_invariance(
        [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "N1CCN(S(=O)(=O)NC(Cc2ccccc2)C(=O)O)CC1",
            "c1ccccc1",
        ],
        sample_size=3,
        randomizations=3,
        seed=1,
    )
    assert result.is_consistent
    assert result.molecules_checked == 3


def test_canonical_encoding_is_stable_for_randomized_smiles():
    smiles = "Cc1nnc(N2CCN(S(=O)(=O)NC(Cc3ccccc3)C(=O)O)CC2)s1"
    codec = FragmentSelfiesCodec()
    expected = codec.encode(smiles, fallback_selfies=False, canonical=True)
    mol = Chem.MolFromSmiles(smiles)

    for _ in range(5):
        randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
        observed = codec.encode(randomized, fallback_selfies=False, canonical=True)
        assert observed == expected


def test_default_encoding_is_not_randomized_by_seed():
    smiles = "Cc1nnc(N2CCN(S(=O)(=O)NC(Cc3ccccc3)C(=O)O)CC2)s1"
    codec = FragmentSelfiesCodec()

    assert codec.encode(smiles, fallback_selfies=False) == codec.encode(
        smiles,
        fallback_selfies=False,
        canonical=False,
        randomized=False,
    )
    assert codec.encode(smiles, fallback_selfies=False, seed=3) == codec.encode(
        smiles,
        fallback_selfies=False,
    )


def test_randomized_encoding_is_seeded_and_roundtrips():
    smiles = "Cc1nnc(N2CCN(S(=O)(=O)NC(Cc3ccccc3)C(=O)O)CC2)s1"
    codec = FragmentSelfiesCodec()

    encoded_variants = {
        codec.encode(smiles, fallback_selfies=False, randomized=True, seed=seed)
        for seed in range(10)
    }
    assert len(encoded_variants) > 1

    expected_smiles = canonical_smiles(smiles)
    for encoded in encoded_variants:
        decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)
        assert decoded == expected_smiles


def test_randomized_fused_aromatic_retries_canonical_fragment_smiles():
    smiles = (
        "c1ccc2[nH]c3c4cccc5c6c(cc(c3nc2c1)c45)"
        "nc1ccccc16"
    )
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=9,
        implicit_probability=0.15,
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert encoded.startswith("[Frag]")
    assert decoded == canonical_smiles(smiles)


def test_randomized_encoding_can_emit_implicit_anchor_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=1,
        implicit_probability=1.0,
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert "][Frag]" in encoded
    assert decoded == canonical_smiles(smiles)


def test_fragment_style_explicit_disables_implicit_anchor_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=1,
        implicit_probability=1.0,
        fragment_style="explicit",
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert "][Frag]" not in encoded
    assert decoded == canonical_smiles(smiles)


def test_fragment_style_implicit_forces_deterministic_anchor_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        canonical=True,
        fragment_style="implicit",
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert "][Frag]" in encoded
    assert decoded == canonical_smiles(smiles)


def test_fragment_style_implicit_can_emit_multiple_anchor_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        canonical=True,
        fragment_style="implicit",
        max_implicit_cuts=2,
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert root_fragment_count(encoded) == 3
    assert decoded == canonical_smiles(smiles)


def test_randomized_auto_can_emit_multiple_implicit_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=0,
        implicit_probability=1.0,
        max_implicit_cuts=2,
    )
    decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)

    assert root_fragment_count(encoded) == 3
    assert decoded == canonical_smiles(smiles)


def test_reserialize_converts_between_explicit_and_implicit_styles():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()
    implicit = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=1,
        fragment_style="implicit",
    )

    explicit = codec.reserialize(implicit, fallback_selfies=False, fragment_style="explicit")
    relinked = codec.reserialize(explicit, fallback_selfies=False, fragment_style="implicit")

    assert "][Frag]" in implicit
    assert "][Frag]" not in explicit
    assert "][Frag]" in relinked
    for encoded in [explicit, relinked]:
        decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)
        assert decoded == canonical_smiles(smiles)


def test_implicit_probability_zero_disables_anchor_roots():
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    codec = FragmentSelfiesCodec()

    encoded = codec.encode(
        smiles,
        fallback_selfies=False,
        randomized=True,
        seed=1,
        implicit_probability=0.0,
    )

    assert "][Frag]" not in encoded


def test_canonical_and_randomize_are_mutually_exclusive():
    codec = FragmentSelfiesCodec()

    with pytest.raises(ValueError, match="mutually exclusive"):
        codec.encode("c1ccccc1", canonical=True, randomized=True)
    with pytest.raises(ValueError, match="implicit_probability"):
        codec.encode("c1ccccc1", randomized=True, implicit_probability=1.5)
    with pytest.raises(ValueError, match="fragment_style"):
        codec.encode("c1ccccc1", fragment_style="unknown")
    with pytest.raises(ValueError, match="max_implicit_cuts"):
        codec.encode("c1ccccc1", max_implicit_cuts=0)


def test_stereochemistry_is_not_retained_by_default():
    vocab, _stats = build_vocabulary_from_smiles(["C[C@H](O)Cl", "C[C@@H](O)Cl"])
    codec = FragmentSelfiesCodec()

    left = codec.encode("C[C@H](O)Cl", fallback_selfies=False, randomized=False)
    right = codec.encode("C[C@@H](O)Cl", fallback_selfies=False, randomized=False)

    assert len(vocab) == 1
    assert left == right
    assert left.startswith("[Frag]")
    assert "[F:" not in left
    decoded = Chem.MolToSmiles(codec.decode(left), canonical=True, isomericSmiles=False)
    assert decoded == "CC(O)Cl"


def test_build_vocab_progress_flags_parse():
    parser = build_parser()
    base_args = ["build-vocab", "--input", "input.smi", "--output", "vocab.jsonl"]

    assert parser.parse_args(base_args).progress is None
    assert parser.parse_args([*base_args, "--progress"]).progress is True
    assert parser.parse_args([*base_args, "--no-progress"]).progress is False
    args = parser.parse_args([*base_args, "--workers", "2", "--chunk-size", "7"])
    assert args.workers == 2
    assert args.chunk_size == 7
    assert args.canonical is False
    assert parser.parse_args([*base_args, "--canonical"]).canonical is True


def test_encode_mode_flags_parse():
    parser = build_parser()
    base_args = ["encode", "--smiles", "c1ccccc1"]

    args = parser.parse_args(base_args)
    assert args.canonical is False
    assert args.randomized is False

    args = parser.parse_args([*base_args, "--canonical"])
    assert args.canonical is True
    assert args.randomized is False

    args = parser.parse_args([*base_args, "--randomized"])
    assert args.canonical is False
    assert args.randomized is True
    assert args.implicit_probability == 0.15
    assert args.max_implicit_cuts == 1
    assert args.fragment_style == "auto"

    args = parser.parse_args(
        [
            *base_args,
            "--randomized",
            "--implicit-probability",
            "0.5",
            "--max-implicit-cuts",
            "3",
            "--style",
            "explicit",
        ]
    )
    assert args.implicit_probability == 0.5
    assert args.max_implicit_cuts == 3
    assert args.fragment_style == "explicit"

    with pytest.raises(SystemExit):
        parser.parse_args([*base_args, "--canonical", "--randomized"])


def test_decode_mode_flags_parse():
    parser = build_parser()
    base_args = ["decode", "--fragment-selfies", "[Frag][C]"]

    args = parser.parse_args(base_args)
    assert args.canonical is False
    assert args.randomized is False
    assert args.repair_missing_return_attachment is False

    args = parser.parse_args([*base_args, "--canonical"])
    assert args.canonical is True
    assert args.randomized is False

    args = parser.parse_args([*base_args, "--randomized"])
    assert args.canonical is False
    assert args.randomized is True

    args = parser.parse_args([*base_args, "--repair-missing-return-attachment"])
    assert args.repair_missing_return_attachment is True

    with pytest.raises(SystemExit):
        parser.parse_args([*base_args, "--canonical", "--randomized"])


def test_reserialize_cli_converts_fragment_styles(capsys):
    codec = FragmentSelfiesCodec()
    implicit = codec.encode(
        "CC(=O)Oc1ccccc1C(=O)O",
        fallback_selfies=False,
        randomized=True,
        seed=1,
        fragment_style="implicit",
    )

    main(["reserialize", "--fragment-selfies", implicit, "--style", "explicit", "--canonical"])
    explicit = capsys.readouterr().out.strip()

    main(["convert", "--fragment-selfies", explicit, "--style", "implicit", "--canonical"])
    relinked = capsys.readouterr().out.strip()

    assert "][Frag]" not in explicit
    assert "][Frag]" in relinked


def test_parallel_vocab_build_matches_single_worker():
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Cc1nnc(N2CCN(S(=O)(=O)NC(Cc3ccccc3)C(=O)O)CC2)s1",
        "c1ccccc1",
        "Cc1c2c(nn1C)CN(C)CC2",
    ]
    sequential_vocab, sequential_stats = build_vocabulary_from_smiles(smiles, num_workers=1, chunk_size=2)
    parallel_vocab, parallel_stats = build_vocabulary_from_smiles(smiles, num_workers=2, chunk_size=2)

    assert parallel_stats == sequential_stats
    assert parallel_vocab.entries == sequential_vocab.entries


def test_build_vocab_progress_writes_to_stderr(tmp_path, capsys):
    input_path = tmp_path / "input.smi"
    output_path = tmp_path / "vocab.jsonl"
    input_path.write_text("c1ccccc1\nCC(=O)O\n", encoding="utf-8")

    main(
        [
            "build-vocab",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--progress",
            "--workers",
            "1",
            "--chunk-size",
            "1",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "Building vocabulary" in captured.err
    assert "2/2 molecules" in captured.err
    assert result["molecules_used"] == 2
    assert output_path.exists()


def test_strict_decode_rejects_unknown_token():
    codec = FragmentSelfiesCodec()
    with pytest.raises(Exception):
        codec.decode("[F:DOES_NOT_EXIST]", strict=True)


def test_decode_defaults_to_non_strict_recovery():
    codec = FragmentSelfiesCodec()
    decoded = Chem.MolToSmiles(codec.decode("not a token"), canonical=True, isomericSmiles=False)
    assert decoded == "C"


def test_non_strict_decode_recovers_from_malformed_generated_fragment():
    codec = FragmentSelfiesCodec()
    malformed = "[Frag][C][Attach:0][Frag@0][C][Dummy][=O]"

    with pytest.raises(Exception):
        codec.decode(malformed, strict=True)

    decoded = Chem.MolToSmiles(codec.decode(malformed, strict=False), canonical=True, isomericSmiles=False)
    assert Chem.MolFromSmiles(decoded) is not None


def test_dot_token_preserves_disconnected_components():
    codec = FragmentSelfiesCodec()
    for strict in [True, False]:
        decoded = Chem.MolToSmiles(
            codec.decode("[Frag][C][.][Frag][C][C][C]", strict=strict),
            canonical=True,
            isomericSmiles=False,
        )
        assert decoded == "C.CCC"


def test_non_strict_decode_links_implicit_anchor_fragments():
    codec = FragmentSelfiesCodec()
    implicit_start = "[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Frag][C][OH0][Dummy]"

    strict_decoded = Chem.MolToSmiles(codec.decode(implicit_start, strict=True), canonical=True, isomericSmiles=False)
    repaired = Chem.MolToSmiles(codec.decode(implicit_start, strict=False), canonical=True, isomericSmiles=False)

    assert strict_decoded == "CO[C]=O"
    assert repaired == strict_decoded
    assert "." not in repaired


def test_non_strict_decode_preserves_implicit_start_atoms_when_linking():
    codec = FragmentSelfiesCodec()
    implicit_start = "[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Frag][C][OH0][Dummy]"

    strict_mol = codec.decode(implicit_start, strict=True)
    repaired_mol = codec.decode(implicit_start, strict=False)
    repaired = Chem.MolToSmiles(repaired_mol, canonical=True, isomericSmiles=False)

    assert repaired_mol.GetNumAtoms() == strict_mol.GetNumAtoms()
    assert "." not in repaired


def test_strict_decode_links_adjacent_root_fragments():
    codec = FragmentSelfiesCodec()
    implicit_start = "[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Frag][C][OH0][Dummy]"

    decoded = Chem.MolToSmiles(codec.decode(implicit_start, strict=True), canonical=True, isomericSmiles=False)

    assert decoded == "CO[C]=O"


def test_strict_decode_connects_explicit_attachment_edge():
    codec = FragmentSelfiesCodec()
    connected = "[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][OH0][Dummy]"

    decoded = Chem.MolToSmiles(codec.decode(connected, strict=True), canonical=True, isomericSmiles=False)

    assert "." not in decoded


def test_decode_repair_missing_return_attachment_selects_farthest_atom():
    codec = FragmentSelfiesCodec()
    missing_return = (
        "[Frag][CH3][Dummy][Frag][CH3][Dummy]"
        "[Attach:0][Frag@0][C][C][C][CH2][Dummy]"
    )

    with pytest.raises(Exception):
        codec.decode(missing_return, strict=True)

    default_recovery = Chem.MolToSmiles(
        codec.decode(missing_return, strict=False),
        canonical=True,
        isomericSmiles=False,
    )
    repaired = Chem.MolToSmiles(
        codec.decode(missing_return, strict=True, repair_missing_return_attachment=True),
        canonical=True,
        isomericSmiles=False,
    )

    assert default_recovery == "CCCCC"
    assert repaired == "CCCCCC"


def test_decode_repair_missing_return_attachment_preserves_linker_seed():
    codec = FragmentSelfiesCodec()
    generated = (
        "[Frag][C][=C][CH0][Branch1][C][Dummy][=C][C][=C][Ring1][#Branch1]"
        "[Frag][N][Branch1][C][C][C][C][NH0][Branch1][C][Dummy][C][Ring1][#Branch1][=O]"
        "[Attach:0][Frag@0][C][C][CH1][Branch1][C][Dummy][C][C][NH0][Ring1][#Branch1][Dummy]"
        "[Attach:1][Frag@1][CH0][Branch1][C][Dummy][=C][C][=C][Branch1][Branch1][C][=C]"
        "[Ring1][#Branch1][CH0][Branch1][C][Dummy][=N][C][=N][Ring1][=Branch2]"
        "[Attach:0][Frag@0][C][=C][Branch1][C][F][C][=C][Branch1][C][F][C][Branch1][C][C]"
        "[=CH0][Ring1][=Branch2][Dummy][pop][pop][pop]"
    )

    with pytest.raises(Exception):
        codec.decode(generated, strict=True)

    default_recovery = codec.decode(generated, strict=False)
    repaired = codec.decode(generated, strict=True, repair_missing_return_attachment=True)

    assert default_recovery.GetNumAtoms() == 32
    assert repaired.GetNumAtoms() == 38
    assert len(Chem.GetMolFrags(repaired, asMols=False, sanitizeFrags=False)) == 1


def test_non_strict_decode_falls_back_to_valid_molecule_for_garbage():
    codec = FragmentSelfiesCodec()
    decoded = Chem.MolToSmiles(codec.decode("not a token", strict=False), canonical=True, isomericSmiles=False)
    assert decoded == "C"


def test_optional_sample_roundtrips_compact_format():
    sample_path = os.environ.get(SAMPLE_SMI_ENV, "")
    if not sample_path:
        pytest.skip(f"set {SAMPLE_SMI_ENV} to run the optional corpus roundtrip test")

    sample_file = Path(sample_path)
    if not sample_file.exists():
        pytest.skip(f"sample file does not exist: {sample_file}")

    codec = FragmentSelfiesCodec()
    checked = 0
    for line in sample_file.read_text(encoding="utf-8").splitlines():
        smiles = line.split()[0] if line.strip() else ""
        if not smiles:
            continue
        expected = canonical_smiles(smiles)
        for encoded in [
            codec.encode(smiles, fallback_selfies=False, canonical=True),
            codec.encode(smiles, fallback_selfies=False, randomized=True, seed=checked),
        ]:
            assert encoded.startswith("[Frag]") or encoded.startswith("[SELFIES]")
            assert "[F:" not in encoded
            decoded = Chem.MolToSmiles(codec.decode(encoded), canonical=True, isomericSmiles=False)
            assert decoded == expected
        checked += 1
        if checked == 50:
            break

    assert checked > 0
