"""Fragment vocabulary utilities."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem


@dataclass(frozen=True)
class FragmentEntry:
    """One fragment token vocabulary entry."""

    token: str
    smiles: str
    count: int = 0
    num_attachments: int = 0


def _clean_dummy_atom(atom: Chem.Atom) -> None:
    atom.SetIsotope(0)
    atom.SetAtomMapNum(0)


def canonicalize_fragment_mol(mol: Chem.Mol) -> str:
    """Return canonical non-isomeric SMILES for a fragment with generic attachment dummies.

    BRICS fragments often contain dummy atoms whose isotopes identify a cut
    bond in a specific molecule. Those labels are useful during encoding, but
    must not become part of the vocabulary key. Stereochemistry is also omitted
    from the default vocabulary representation.
    """

    clone = Chem.Mol(mol)
    for atom in clone.GetAtoms():
        if atom.GetAtomicNum() == 0:
            _clean_dummy_atom(atom)
    return Chem.MolToSmiles(clone, canonical=True, isomericSmiles=False)


def canonicalize_fragment_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid fragment SMILES: {smiles}")
    return canonicalize_fragment_mol(mol)


def count_attachments(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid fragment SMILES: {smiles}")
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


class FragmentVocabulary:
    """Bidirectional mapping between fragment SMILES and compact tokens."""

    def __init__(self, entries: Iterable[FragmentEntry] = ()):
        self._by_token: dict[str, FragmentEntry] = {}
        self._by_smiles: dict[str, FragmentEntry] = {}
        for entry in entries:
            self.add_entry(entry)

    def __len__(self) -> int:
        return len(self._by_token)

    def __contains__(self, value: str) -> bool:
        return value in self._by_token or value in self._by_smiles

    @property
    def entries(self) -> tuple[FragmentEntry, ...]:
        return tuple(sorted(self._by_token.values(), key=lambda e: e.token))

    def add_entry(self, entry: FragmentEntry) -> None:
        if entry.token in self._by_token:
            raise ValueError(f"duplicate fragment token: {entry.token}")
        if entry.smiles in self._by_smiles:
            raise ValueError(f"duplicate fragment SMILES: {entry.smiles}")
        self._by_token[entry.token] = entry
        self._by_smiles[entry.smiles] = entry

    def token_for_smiles(self, smiles: str) -> str | None:
        canonical = canonicalize_fragment_smiles(smiles)
        entry = self._by_smiles.get(canonical)
        return None if entry is None else entry.token

    def entry_for_token(self, token: str) -> FragmentEntry | None:
        return self._by_token.get(token)

    def entry_for_smiles(self, smiles: str) -> FragmentEntry | None:
        canonical = canonicalize_fragment_smiles(smiles)
        return self._by_smiles.get(canonical)

    @classmethod
    def from_counter(
        cls,
        counter: Counter[str],
        *,
        min_count: int = 1,
        max_fragments: int | None = None,
        token_prefix: str = "F",
    ) -> "FragmentVocabulary":
        """Create a stable count-sorted vocabulary from fragment counts."""

        items = [item for item in counter.items() if item[1] >= min_count]
        items.sort(key=lambda item: (-item[1], item[0]))
        if max_fragments is not None:
            items = items[:max_fragments]

        width = max(6, len(str(len(items))))
        entries = []
        for idx, (smiles, count) in enumerate(items, start=1):
            entries.append(
                FragmentEntry(
                    token=f"{token_prefix}{idx:0{width}d}",
                    smiles=smiles,
                    count=count,
                    num_attachments=count_attachments(smiles),
                )
            )
        return cls(entries)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "FragmentVocabulary":
        entries = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(FragmentEntry(**data))
        return cls(entries)

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), sort_keys=True))
                handle.write("\n")
