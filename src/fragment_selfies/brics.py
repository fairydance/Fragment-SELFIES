"""BRICS fragmentation and vocabulary construction."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from rdkit import Chem
from rdkit.Chem import BRICS

from .vocabulary import FragmentVocabulary, canonicalize_fragment_mol


def smiles_from_line(line: str) -> str | None:
    """Extract the first whitespace-delimited field from an SMI line."""

    stripped = line.strip()
    if not stripped:
        return None
    return stripped.split()[0]


def iter_smiles_file(path: str | Path) -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            smiles = smiles_from_line(line)
            if smiles is not None:
                yield smiles


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def brics_bond_indices(mol: Chem.Mol) -> list[int]:
    """Return RDKit bond indices selected by BRICS."""

    indices = []
    for atom_pair, _labels in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(*atom_pair)
        if bond is not None:
            indices.append(bond.GetIdx())
    return sorted(set(indices))


def fragment_mol_with_brics(mol: Chem.Mol, *, allow_empty: bool = True) -> list[Chem.Mol]:
    """Split a molecule with BRICS and return RDKit fragment molecules.

    Cut-bond dummy labels are intentionally molecule-local; downstream
    canonicalization removes them for vocabulary construction.
    """

    bond_indices = brics_bond_indices(mol)
    if not bond_indices:
        return [Chem.Mol(mol)] if allow_empty else []
    labels = [(idx + 1, idx + 1) for idx in range(len(bond_indices))]
    fragmented = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True, dummyLabels=labels)
    return list(Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True))


def brics_fragment_smiles(smiles: str, *, canonical_input: bool = True) -> list[str]:
    """Return canonical generic BRICS fragment SMILES for one molecule."""

    if canonical_input:
        smiles = canonicalize_smiles(smiles) or smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    fragments = fragment_mol_with_brics(mol, allow_empty=True)
    return sorted(canonicalize_fragment_mol(fragment) for fragment in fragments)


@dataclass
class VocabularyBuildStats:
    molecules_seen: int = 0
    molecules_used: int = 0
    invalid_smiles: int = 0
    unique_fragments: int = 0


ProgressCallback = Callable[[int], None]


def _iter_smiles_chunks(
    smiles_iter: Iterable[str],
    *,
    max_molecules: int | None,
    chunk_size: int,
) -> Iterator[list[str]]:
    chunk = []
    molecules_seen = 0
    for smiles in smiles_iter:
        if max_molecules is not None and molecules_seen >= max_molecules:
            break
        chunk.append(smiles)
        molecules_seen += 1
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _process_smiles_chunk(smiles_chunk: list[str], canonical_input: bool) -> tuple[Counter[str], VocabularyBuildStats]:
    counter: Counter[str] = Counter()
    stats = VocabularyBuildStats()

    for smiles in smiles_chunk:
        stats.molecules_seen += 1
        if canonical_input:
            smiles = canonicalize_smiles(smiles) or ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            stats.invalid_smiles += 1
            continue
        for fragment in fragment_mol_with_brics(mol, allow_empty=True):
            counter[canonicalize_fragment_mol(fragment)] += 1
        stats.molecules_used += 1

    stats.unique_fragments = len(counter)
    return counter, stats


def _merge_build_stats(total: VocabularyBuildStats, partial: VocabularyBuildStats) -> None:
    total.molecules_seen += partial.molecules_seen
    total.molecules_used += partial.molecules_used
    total.invalid_smiles += partial.invalid_smiles


def build_fragment_counter(
    smiles_iter: Iterable[str],
    *,
    canonical_input: bool = True,
    max_molecules: int | None = None,
    num_workers: int = 1,
    chunk_size: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Counter[str], VocabularyBuildStats]:
    """Stream molecules and count canonical BRICS fragments."""

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    counter: Counter[str] = Counter()
    stats = VocabularyBuildStats()
    chunks = _iter_smiles_chunks(smiles_iter, max_molecules=max_molecules, chunk_size=chunk_size)

    if num_workers == 1:
        for chunk in chunks:
            chunk_counter, chunk_stats = _process_smiles_chunk(chunk, canonical_input)
            counter.update(chunk_counter)
            _merge_build_stats(stats, chunk_stats)
            if progress_callback is not None:
                progress_callback(chunk_stats.molecules_seen)
    else:
        max_pending = num_workers * 2
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pending = set()

            def submit_next() -> bool:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    return False
                pending.add(executor.submit(_process_smiles_chunk, chunk, canonical_input))
                return True

            for _ in range(max_pending):
                if not submit_next():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    chunk_counter, chunk_stats = future.result()
                    counter.update(chunk_counter)
                    _merge_build_stats(stats, chunk_stats)
                    if progress_callback is not None:
                        progress_callback(chunk_stats.molecules_seen)
                    submit_next()

    stats.unique_fragments = len(counter)
    return counter, stats


def build_vocabulary_from_smiles(
    smiles_iter: Iterable[str],
    *,
    canonical_input: bool = True,
    max_molecules: int | None = None,
    min_count: int = 1,
    max_fragments: int | None = None,
    num_workers: int = 1,
    chunk_size: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> tuple[FragmentVocabulary, VocabularyBuildStats]:
    counter, stats = build_fragment_counter(
        smiles_iter,
        canonical_input=canonical_input,
        max_molecules=max_molecules,
        num_workers=num_workers,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
    )
    vocab = FragmentVocabulary.from_counter(
        counter,
        min_count=min_count,
        max_fragments=max_fragments,
    )
    return vocab, stats


@dataclass
class BRICSInvarianceResult:
    molecules_checked: int = 0
    invalid_smiles: int = 0
    mismatches: list[dict[str, object]] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return not self.mismatches


def verify_brics_fragment_invariance(
    smiles_iter: Iterable[str],
    *,
    sample_size: int = 1000,
    randomizations: int = 5,
    seed: int = 0,
) -> BRICSInvarianceResult:
    """Check if randomized SMILES forms yield identical BRICS fragments.

    This is an empirical verification helper. It compares fragment multisets
    from canonical and randomized SMILES generated by RDKit for the same mol.
    """

    rng = __import__("random").Random(seed)
    reservoir: list[str] = []
    seen = 0
    for smiles in smiles_iter:
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(smiles)
        else:
            j = rng.randrange(seen)
            if j < sample_size:
                reservoir[j] = smiles

    result = BRICSInvarianceResult()
    for smiles in reservoir:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.invalid_smiles += 1
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        baseline = Counter(brics_fragment_smiles(canonical, canonical_input=False))
        result.molecules_checked += 1

        for _ in range(randomizations):
            randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
            observed = Counter(brics_fragment_smiles(randomized, canonical_input=False))
            if observed != baseline:
                result.mismatches.append(
                    {
                        "input": smiles,
                        "canonical": canonical,
                        "randomized": randomized,
                        "baseline": dict(baseline),
                        "observed": dict(observed),
                    }
                )
                break
    return result
