"""Fragment-SELFIES encoder and decoder."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

import selfies as sf
from rdkit import Chem
from rdkit.Chem import BRICS

from .exceptions import FragmentSelfiesDecodeError, FragmentSelfiesEncodeError
from .tokens import (
    DOT_TOKEN,
    POP_TOKEN,
    SELFIES_END,
    SELFIES_START,
    make_attachment_token,
    make_fragment_token,
    parse_attachment_token,
    parse_fragment_token,
    split_tokens,
)
from .vocabulary import FragmentVocabulary


# The sentinel is only used inside SELFIES encoder/decoder calls; serialized
# Fragment-SELFIES uses non-atomic Dummy tokens instead.
ATTACHMENT_ATOMIC_NUM = 116
ATTACHMENT_SYMBOL = "Lv"
DUMMY_SYMBOL = "Dummy"
DUMMY_TOKEN_RE = re.compile(
    r"^\[(?P<prefix>[=#/\\]?)(?P<symbol>Dummy|Lv)(?P<suffix>[^\]]*)\]$"
)


@dataclass(frozen=True)
class Attachment:
    placeholder_idx: int
    parent_idx: int
    bond_type: Chem.BondType


@dataclass(frozen=True)
class FragmentTemplate:
    smiles: str
    mol: Chem.Mol
    attachments: tuple[Attachment, ...]
    core_atom_indices: tuple[int, ...]

    @property
    def heavy_atom_count(self) -> int:
        return len(self.core_atom_indices)

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> "FragmentTemplate":
        mol = _attachment_placeholder_mol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        placeholder_indices = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == ATTACHMENT_ATOMIC_NUM
        ]
        placeholder_indices.sort(key=lambda idx: (ranks[idx], idx))

        attachments = []
        for idx in placeholder_indices:
            atom = mol.GetAtomWithIdx(idx)
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                raise ValueError(f"attachment placeholder must have exactly one neighbor: {smiles}")
            parent = neighbors[0]
            bond = mol.GetBondBetweenAtoms(idx, parent.GetIdx())
            attachments.append(Attachment(idx, parent.GetIdx(), bond.GetBondType()))

        core_atoms = tuple(
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != ATTACHMENT_ATOMIC_NUM
        )
        return cls(smiles, mol, tuple(attachments), core_atoms)

    @classmethod
    def from_selfies_tokens(cls, tokens: list[str]) -> "FragmentTemplate":
        if not tokens:
            raise FragmentSelfiesDecodeError("fragment is missing SELFIES tokens")
        tokens = _deserialize_dummy_tokens(tokens)
        try:
            smiles = sf.decoder("".join(tokens))
        except Exception as exc:
            raise FragmentSelfiesDecodeError("failed to decode fragment SELFIES") from exc
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FragmentSelfiesDecodeError(f"invalid fragment SMILES decoded from SELFIES: {smiles}")
        try:
            return cls.from_mol(mol)
        except ValueError as exc:
            raise FragmentSelfiesDecodeError(str(exc)) from exc


@dataclass
class _FragmentInstance:
    selfies_tokens: list[str]
    template: FragmentTemplate
    conn_to_attachment: dict[int, int]

    @property
    def sort_key(self) -> str:
        return "".join(self.selfies_tokens)


@dataclass
class _PlacedFragment:
    template: FragmentTemplate
    atom_map: dict[int, int]
    used_attachments: set[int]
    group_id: int


DEFAULT_IMPLICIT_PROBABILITY = 0.15
DEFAULT_MAX_IMPLICIT_CUTS = 1
FRAGMENT_STYLE_AUTO = "auto"
FRAGMENT_STYLE_EXPLICIT = "explicit"
FRAGMENT_STYLE_IMPLICIT = "implicit"
FRAGMENT_STYLES = {
    FRAGMENT_STYLE_AUTO,
    FRAGMENT_STYLE_EXPLICIT,
    FRAGMENT_STYLE_IMPLICIT,
}


def _copy_atom(atom: Chem.Atom) -> Chem.Atom:
    copied = Chem.Atom(atom)
    copied.SetAtomMapNum(0)
    return copied


def _randomize_mol(mol: Chem.Mol, rng: random.Random) -> Chem.Mol:
    atom_indices = list(range(mol.GetNumAtoms()))
    rng.shuffle(atom_indices)
    return Chem.RenumberAtoms(mol, atom_indices)


def _contains_attachment_sentinel(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == ATTACHMENT_ATOMIC_NUM for atom in mol.GetAtoms())


def _brics_bond_indices(mol: Chem.Mol) -> list[int]:
    bond_indices = []
    for atom_pair, _labels in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(*atom_pair)
        if bond is not None:
            bond_indices.append(bond.GetIdx())
    return sorted(set(bond_indices))


def _fragment_with_connection_labels(mol: Chem.Mol) -> list[Chem.Mol]:
    bond_indices = _brics_bond_indices(mol)
    if not bond_indices:
        return [Chem.Mol(mol)]
    labels = [(idx + 1, idx + 1) for idx in range(len(bond_indices))]
    fragmented = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True, dummyLabels=labels)
    return list(Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True))


def _attachment_placeholder_mol(mol: Chem.Mol) -> Chem.Mol:
    clone = Chem.Mol(mol)
    for atom in clone.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(ATTACHMENT_ATOMIC_NUM)
        if atom.GetAtomicNum() == ATTACHMENT_ATOMIC_NUM:
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(clone)
    except Exception as exc:
        raise FragmentSelfiesEncodeError("failed to sanitize fragment attachment placeholders") from exc
    return clone


def _attachment_placeholder_to_dummy_mol(mol: Chem.Mol) -> Chem.Mol:
    clone = Chem.Mol(mol)
    for atom in clone.GetAtoms():
        if atom.GetAtomicNum() == ATTACHMENT_ATOMIC_NUM:
            atom.SetAtomicNum(0)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(clone)
        Chem.AssignStereochemistry(clone, force=True)
    except Exception as exc:
        raise FragmentSelfiesDecodeError("decoded fragment failed RDKit sanitization") from exc
    return clone


def _attachment_order_for_labeled_fragment(fragment: Chem.Mol) -> dict[int, int]:
    generic = _attachment_placeholder_mol(fragment)
    ranks = list(Chem.CanonicalRankAtoms(generic, breakTies=True))
    placeholder_indices = [
        atom.GetIdx()
        for atom in generic.GetAtoms()
        if atom.GetAtomicNum() == ATTACHMENT_ATOMIC_NUM
    ]
    placeholder_indices.sort(key=lambda idx: (ranks[idx], idx))
    return {idx: pos for pos, idx in enumerate(placeholder_indices)}


def _fragment_to_selfies_tokens(fragment: Chem.Mol, *, canonical: bool) -> list[str]:
    placeholder_mol = _attachment_placeholder_mol(fragment)
    smiles_candidates = [
        Chem.MolToSmiles(placeholder_mol, canonical=canonical, isomericSmiles=False)
    ]
    if not canonical:
        canonical_smiles = Chem.MolToSmiles(
            placeholder_mol,
            canonical=True,
            isomericSmiles=False,
        )
        if canonical_smiles != smiles_candidates[0]:
            smiles_candidates.append(canonical_smiles)

    last_exc = None
    for smiles in smiles_candidates:
        try:
            fragment_selfies = sf.encoder(smiles)
            return _serialize_dummy_tokens(split_tokens(fragment_selfies))
        except Exception as exc:
            last_exc = exc
    raise FragmentSelfiesEncodeError(
        f"failed to encode fragment as SELFIES: {smiles_candidates[0]}"
    ) from last_exc


def _translate_dummy_tokens(tokens: list[str], *, source: str, target: str) -> list[str]:
    translated = []
    for token in tokens:
        match = DUMMY_TOKEN_RE.match(token)
        if match is not None and match.group("symbol") == source:
            translated.append(f"[{match.group('prefix')}{target}{match.group('suffix')}]")
        else:
            translated.append(token)
    return translated


def _serialize_dummy_tokens(tokens: list[str]) -> list[str]:
    return _translate_dummy_tokens(tokens, source=ATTACHMENT_SYMBOL, target=DUMMY_SYMBOL)


def _deserialize_dummy_tokens(tokens: list[str]) -> list[str]:
    return _translate_dummy_tokens(tokens, source=DUMMY_SYMBOL, target=ATTACHMENT_SYMBOL)


def _fragment_instances(mol: Chem.Mol, *, canonical_fragments: bool) -> list[_FragmentInstance]:
    instances = []
    for fragment in _fragment_with_connection_labels(mol):
        selfies_tokens = _fragment_to_selfies_tokens(fragment, canonical=canonical_fragments)
        dummy_to_attachment = _attachment_order_for_labeled_fragment(fragment)
        conn_to_attachment = {}
        for atom in fragment.GetAtoms():
            if atom.GetAtomicNum() == 0 and atom.GetIsotope() > 0:
                conn_to_attachment[atom.GetIsotope()] = dummy_to_attachment[atom.GetIdx()]
        instances.append(
            _FragmentInstance(
                selfies_tokens=selfies_tokens,
                template=FragmentTemplate.from_selfies_tokens(selfies_tokens),
                conn_to_attachment=conn_to_attachment,
            )
        )
    return instances


def _is_control_token(symbol: str) -> bool:
    return (
        symbol in {POP_TOKEN, DOT_TOKEN, SELFIES_START, SELFIES_END}
        or parse_attachment_token(symbol) is not None
        or parse_fragment_token(symbol) is not None
    )


def _edge_key(left_idx: int, right_idx: int) -> tuple[int, int]:
    return (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)


def _connected_components(
    adjacency: dict[int, list[tuple[int, int, int]]],
    count: int,
) -> dict[int, int]:
    components = {}
    component_idx = 0
    for root in range(count):
        if root in components:
            continue
        stack = [root]
        components[root] = component_idx
        while stack:
            current = stack.pop()
            for neighbor, _parent_att, _child_att in adjacency[current]:
                if neighbor in components:
                    continue
                components[neighbor] = component_idx
                stack.append(neighbor)
        component_idx += 1
    return components


def _normalized_fragment_style(fragment_style: str) -> str:
    if fragment_style not in FRAGMENT_STYLES:
        allowed = ", ".join(sorted(FRAGMENT_STYLES))
        raise ValueError(f"fragment_style must be one of: {allowed}")
    return fragment_style


def _validated_max_implicit_cuts(max_implicit_cuts: int) -> int:
    if max_implicit_cuts < 1:
        raise ValueError("max_implicit_cuts must be at least 1")
    return max_implicit_cuts


def _select_implicit_cut_edges(
    component_edges: dict[int, list[tuple[int, int]]],
    *,
    fragment_style: str,
    implicit_probability: float,
    max_implicit_cuts: int,
    rng: random.Random | None,
    edge_key,
) -> set[tuple[int, int]]:
    cut_edges = set()
    for edges in component_edges.values():
        if not edges:
            continue
        cut_limit = min(max_implicit_cuts, len(edges))
        if fragment_style == FRAGMENT_STYLE_IMPLICIT:
            cut_count = cut_limit
        elif fragment_style == FRAGMENT_STYLE_AUTO and rng is not None:
            if implicit_probability <= 0.0 or rng.random() >= implicit_probability:
                continue
            cut_count = rng.randint(1, cut_limit)
        else:
            continue

        if rng is None:
            selected_edges = sorted(edges, key=edge_key)[:cut_count]
        else:
            selected_edges = rng.sample(edges, cut_count)
        cut_edges.update(_edge_key(*edge) for edge in selected_edges)
    return cut_edges


class FragmentSelfiesCodec:
    """Encode molecules as BRICS-fragmented runs of standard SELFIES tokens."""

    def __init__(self, vocabulary: FragmentVocabulary | None = None):
        self.vocabulary = vocabulary

    def encode_fragment(
        self,
        fragment_mol_or_smiles: Chem.Mol | str,
        *,
        canonical: bool = False,
        randomized: bool = False,
        seed: int | None = None,
        include_fragment_token: bool = True,
    ) -> str:
        if canonical and randomized:
            raise ValueError("canonical=True and randomized=True are mutually exclusive")

        mol = (
            Chem.MolFromSmiles(fragment_mol_or_smiles)
            if isinstance(fragment_mol_or_smiles, str)
            else Chem.Mol(fragment_mol_or_smiles)
        )
        if mol is None:
            raise FragmentSelfiesEncodeError(f"invalid fragment: {fragment_mol_or_smiles}")
        if _contains_attachment_sentinel(mol):
            raise FragmentSelfiesEncodeError(
                f"{ATTACHMENT_SYMBOL} is reserved as the internal Dummy attachment sentinel"
            )
        if len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) != 1:
            raise FragmentSelfiesEncodeError("fragment SMILES must contain exactly one component")

        rng = random.Random(seed) if randomized else None
        if rng is not None:
            mol = _randomize_mol(mol, rng)
        elif canonical:
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise FragmentSelfiesEncodeError(f"invalid canonical fragment: {smiles}")

        tokens = _fragment_to_selfies_tokens(mol, canonical=canonical)
        prefix = make_fragment_token() if include_fragment_token else ""
        return prefix + "".join(tokens)

    def decode_fragment(self, fragment_selfies: str) -> Chem.Mol:
        try:
            tokens = split_tokens(fragment_selfies)
        except ValueError as exc:
            raise FragmentSelfiesDecodeError(str(exc)) from exc
        if not tokens:
            raise FragmentSelfiesDecodeError("fragment is empty")

        fragment_token = parse_fragment_token(tokens[0])
        if fragment_token is not None:
            tokens = tokens[1:]
        if not tokens:
            raise FragmentSelfiesDecodeError("fragment is missing SELFIES tokens")
        for token in tokens:
            if _is_control_token(token):
                raise FragmentSelfiesDecodeError(f"unexpected structural token in fragment: {token}")

        template = FragmentTemplate.from_selfies_tokens(tokens)
        return _attachment_placeholder_to_dummy_mol(template.mol)

    def encode(
        self,
        mol_or_smiles: Chem.Mol | str,
        *,
        fallback_selfies: bool = True,
        canonical: bool = False,
        randomized: bool = False,
        seed: int | None = None,
        implicit_probability: float = DEFAULT_IMPLICIT_PROBABILITY,
        max_implicit_cuts: int = DEFAULT_MAX_IMPLICIT_CUTS,
        fragment_style: str = FRAGMENT_STYLE_AUTO,
    ) -> str:
        if canonical and randomized:
            raise ValueError("canonical=True and randomized=True are mutually exclusive")
        if not 0.0 <= implicit_probability <= 1.0:
            raise ValueError("implicit_probability must be between 0.0 and 1.0")
        fragment_style = _normalized_fragment_style(fragment_style)
        max_implicit_cuts = _validated_max_implicit_cuts(max_implicit_cuts)

        mol = (
            Chem.MolFromSmiles(mol_or_smiles)
            if isinstance(mol_or_smiles, str)
            else Chem.Mol(mol_or_smiles)
        )
        if mol is None:
            raise FragmentSelfiesEncodeError(f"invalid molecule: {mol_or_smiles}")

        rng = random.Random(seed) if randomized else None
        if rng is not None:
            mol = _randomize_mol(mol, rng)
            fallback_smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)
        elif canonical:
            fallback_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            mol = Chem.MolFromSmiles(fallback_smiles)
            if mol is None:
                raise FragmentSelfiesEncodeError(f"invalid canonical molecule: {fallback_smiles}")
        else:
            fallback_smiles = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)

        if _contains_attachment_sentinel(mol):
            if fallback_selfies:
                return SELFIES_START + sf.encoder(fallback_smiles) + SELFIES_END
            raise FragmentSelfiesEncodeError(
                f"{ATTACHMENT_SYMBOL} is reserved as the internal Dummy attachment sentinel"
            )

        try:
            instances = _fragment_instances(mol, canonical_fragments=canonical)
        except FragmentSelfiesEncodeError:
            if fallback_selfies:
                return SELFIES_START + sf.encoder(fallback_smiles) + SELFIES_END
            raise

        if not instances:
            return ""

        conn_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for frag_idx, instance in enumerate(instances):
            for conn_id, attachment_idx in instance.conn_to_attachment.items():
                conn_edges[conn_id].append((frag_idx, attachment_idx))

        adjacency: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for endpoints in conn_edges.values():
            if len(endpoints) != 2:
                continue
            (left_idx, left_att), (right_idx, right_att) = endpoints
            adjacency[left_idx].append((right_idx, left_att, right_att))
            adjacency[right_idx].append((left_idx, right_att, left_att))
        components = _connected_components(adjacency, len(instances))

        def root_key(idx: int) -> tuple[int, str]:
            inst = instances[idx]
            return (inst.template.heavy_atom_count, inst.sort_key)

        def edge_key_for_selection(
            edge: tuple[int, int]
        ) -> tuple[tuple[int, str], tuple[int, str], int, int]:
            left_idx, right_idx = _edge_key(*edge)
            left_key = root_key(left_idx)
            right_key = root_key(right_idx)
            if right_key < left_key:
                left_key, right_key = right_key, left_key
            return (left_key, right_key, left_idx, right_idx)

        component_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for left_idx, edges in adjacency.items():
            for right_idx, _left_att, _right_att in edges:
                if left_idx < right_idx:
                    component_edges[components[left_idx]].append((left_idx, right_idx))

        cut_edges = _select_implicit_cut_edges(
            component_edges,
            fragment_style=fragment_style,
            implicit_probability=implicit_probability,
            max_implicit_cuts=max_implicit_cuts,
            rng=rng,
            edge_key=edge_key_for_selection,
        )

        tokens: list[str] = []
        visited: set[int] = set()
        emitted_components: set[int] = set()

        def emit_fragment(idx: int, child_attachment: int | None = None) -> None:
            tokens.append(make_fragment_token(child_attachment))
            tokens.extend(instances[idx].selfies_tokens)

        def emit_node(idx: int) -> None:
            visited.add(idx)
            children = [edge for edge in adjacency[idx] if edge[0] not in visited]
            if rng is None:
                children.sort(key=lambda edge: (edge[1], instances[edge[0]].sort_key, edge[2]))
            else:
                rng.shuffle(children)
            for child_idx, parent_att, child_att in children:
                if _edge_key(idx, child_idx) in cut_edges:
                    continue
                tokens.append(make_attachment_token(parent_att))
                emit_fragment(child_idx, child_att)
                emit_node(child_idx)
                tokens.append(POP_TOKEN)

        roots = list(range(len(instances)))
        if rng is None:
            roots.sort(key=root_key, reverse=True)
        else:
            rng.shuffle(roots)
        for root in roots:
            if root in visited:
                continue
            if tokens and components[root] not in emitted_components:
                tokens.append(DOT_TOKEN)
            emit_fragment(root)
            emit_node(root)
            emitted_components.add(components[root])

        return "".join(tokens)

    def reserialize(
        self,
        fragment_selfies: str,
        *,
        fallback_selfies: bool = True,
        canonical: bool = True,
        randomized: bool = False,
        seed: int | None = None,
        implicit_probability: float = DEFAULT_IMPLICIT_PROBABILITY,
        max_implicit_cuts: int = DEFAULT_MAX_IMPLICIT_CUTS,
        fragment_style: str = FRAGMENT_STYLE_EXPLICIT,
        strict: bool = False,
        repair_missing_return_attachment: bool = False,
    ) -> str:
        """Decode and re-encode Fragment-SELFIES in the requested style."""
        mol = self.decode(
            fragment_selfies,
            strict=strict,
            repair_missing_return_attachment=repair_missing_return_attachment,
        )
        return self.encode(
            mol,
            fallback_selfies=fallback_selfies,
            canonical=canonical,
            randomized=randomized,
            seed=seed,
            implicit_probability=implicit_probability,
            max_implicit_cuts=max_implicit_cuts,
            fragment_style=fragment_style,
        )

    def decode(
        self,
        fragment_selfies: str,
        *,
        strict: bool = False,
        repair_missing_return_attachment: bool = False,
    ) -> Chem.Mol:
        try:
            tokens = split_tokens(fragment_selfies)
        except ValueError as exc:
            if strict:
                raise FragmentSelfiesDecodeError(str(exc)) from exc
            tokens = re.findall(r"\[[^\]]+\]", fragment_selfies)

        builder = _MoleculeBuilder(
            strict=strict,
            repair_missing_return_attachment=repair_missing_return_attachment,
        )
        idx = 0
        while idx < len(tokens):
            symbol = tokens[idx]
            if symbol == SELFIES_START:
                selfies_tokens = []
                idx += 1
                while idx < len(tokens) and tokens[idx] != SELFIES_END:
                    selfies_tokens.append(tokens[idx])
                    idx += 1
                if idx == len(tokens):
                    if strict:
                        raise FragmentSelfiesDecodeError("unterminated SELFIES fallback block")
                    idx -= 1
                try:
                    smiles = sf.decoder("".join(selfies_tokens))
                    builder.add_disconnected_mol(Chem.MolFromSmiles(smiles))
                except Exception as exc:
                    if strict:
                        raise FragmentSelfiesDecodeError("failed to decode SELFIES fallback block") from exc
                idx += 1
                continue

            fragment_token = parse_fragment_token(symbol)
            if fragment_token is not None:
                fragment_tokens = []
                idx += 1
                while idx < len(tokens) and not _is_control_token(tokens[idx]):
                    fragment_tokens.append(tokens[idx])
                    idx += 1
                builder.add_fragment(fragment_tokens, fragment_token.attachment_index)
                continue

            builder.consume(symbol)
            idx += 1
        return builder.finish()


class _MoleculeBuilder:
    def __init__(
        self,
        *,
        strict: bool,
        repair_missing_return_attachment: bool = False,
    ):
        self.strict = strict
        self.repair_missing_return_attachment = repair_missing_return_attachment
        self.mol = Chem.RWMol()
        self.placed: list[_PlacedFragment] = []
        self.current: int | None = None
        self.pending_attachment: int | None = None
        self.stack: list[int | None] = []
        self.current_group_id = 0

    def consume(self, symbol: str) -> None:
        if symbol == POP_TOKEN:
            self.current = self.stack.pop() if self.stack else self.current
            self.pending_attachment = None
            return
        if symbol == DOT_TOKEN:
            self.current = None
            self.pending_attachment = None
            self.stack.clear()
            self.current_group_id += 1
            return
        if (attachment_idx := parse_attachment_token(symbol)) is not None:
            self.pending_attachment = self._next_available_attachment(self.current, attachment_idx)
            return
        if self.strict:
            raise FragmentSelfiesDecodeError(f"unknown token outside fragment: {symbol}")

    def add_fragment(self, fragment_tokens: list[str], child_attachment: int | None) -> None:
        try:
            template = FragmentTemplate.from_selfies_tokens(fragment_tokens)
        except FragmentSelfiesDecodeError:
            if self.strict:
                raise
            self.pending_attachment = None
            return
        self._place_fragment(template, child_attachment)

    def add_disconnected_mol(self, mol: Chem.Mol | None) -> None:
        if mol is None:
            return
        atom_map = {}
        for atom in mol.GetAtoms():
            atom_map[atom.GetIdx()] = self.mol.AddAtom(_copy_atom(atom))
        for bond in mol.GetBonds():
            self.mol.AddBond(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()], bond.GetBondType())
        self.current = None
        self.pending_attachment = None

    def _next_available_attachment(self, placed_idx: int | None, requested: int | None) -> int | None:
        if placed_idx is None:
            return None
        placed = self.placed[placed_idx]
        total = len(placed.template.attachments)
        if total == 0:
            return None
        start = 0 if requested is None else requested % total
        for offset in range(total):
            candidate = (start + offset) % total
            if candidate not in placed.used_attachments:
                return candidate
        return None

    def _copy_template_core(self, template: FragmentTemplate) -> dict[int, int]:
        atom_map = {}
        for atom_idx in template.core_atom_indices:
            atom = template.mol.GetAtomWithIdx(atom_idx)
            atom_map[atom_idx] = self.mol.AddAtom(_copy_atom(atom))
        for bond in template.mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in atom_map and end in atom_map:
                self.mol.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
        return atom_map

    def _place_fragment(self, template: FragmentTemplate, child_attachment: int | None) -> None:
        parent_idx = self.current
        parent_attachment = self.pending_attachment
        if parent_idx is not None and parent_attachment is None and child_attachment is not None:
            parent_attachment = self._next_available_attachment(parent_idx, None)

        atom_map = self._copy_template_core(template)
        placed_idx = len(self.placed)
        group_id = self.current_group_id
        if parent_idx is not None and parent_attachment is not None:
            group_id = self.placed[parent_idx].group_id
        placed = _PlacedFragment(template=template, atom_map=atom_map, used_attachments=set(), group_id=group_id)
        self.placed.append(placed)

        if parent_idx is None or parent_attachment is None:
            self.current = placed_idx
            self.pending_attachment = None
            return

        child_attachment = self._next_available_attachment(placed_idx, child_attachment)
        if child_attachment is None:
            self.current = parent_idx
            self.pending_attachment = None
            return

        parent = self.placed[parent_idx]
        parent_att = parent.template.attachments[parent_attachment]
        child_att = placed.template.attachments[child_attachment]
        parent_atom = parent.atom_map[parent_att.parent_idx]
        child_atom = placed.atom_map[child_att.parent_idx]
        bond_type = parent_att.bond_type if parent_att.bond_type == child_att.bond_type else Chem.BondType.SINGLE

        if self.mol.GetBondBetweenAtoms(parent_atom, child_atom) is None:
            self.mol.AddBond(parent_atom, child_atom, bond_type)
            parent.used_attachments.add(parent_attachment)
            placed.used_attachments.add(child_attachment)
            self.stack.append(parent_idx)
            self.current = placed_idx
        else:
            self.current = parent_idx
        self.pending_attachment = None

    def finish(self) -> Chem.Mol:
        mol = self.mol.GetMol()
        if mol.GetNumAtoms() == 0:
            if self.strict:
                return mol
            return _fallback_valid_mol()
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragments) > 1:
            linked = self._link_components_through_unused_attachments(mol)
            if linked is not None:
                return linked
            if self.repair_missing_return_attachment:
                repaired = self._repair_missing_return_attachment(mol)
                if repaired is not None:
                    return repaired
            if self.strict:
                raise FragmentSelfiesDecodeError("disconnected implicit anchors could not be connected")
            return _largest_component_mol(mol)
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, force=True)
        except Exception as exc:
            if not self.strict:
                return self._best_effort_mol()
            raise FragmentSelfiesDecodeError("decoded molecule failed RDKit sanitization") from exc
        return mol

    def _available_attachment_sites(self) -> list[dict[str, object]]:
        sites = []
        for placed_idx, placed in enumerate(self.placed):
            for attachment_idx, attachment in enumerate(placed.template.attachments):
                if attachment_idx in placed.used_attachments:
                    continue
                parent_atom = placed.atom_map.get(attachment.parent_idx)
                if parent_atom is None:
                    continue
                sites.append(
                    {
                        "placed_idx": placed_idx,
                        "attachment_idx": attachment_idx,
                        "atom_idx": parent_atom,
                        "bond_type": attachment.bond_type,
                        "group_id": placed.group_id,
                    }
                )
        return sites

    def _linked_or_largest_component_mol(self, mol: Chem.Mol | None = None) -> Chem.Mol:
        if mol is None:
            mol = self.mol.GetMol()
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragments) <= 1:
            return mol

        linked = self._link_components_through_unused_attachments(mol)
        if linked is not None:
            return linked
        if self.repair_missing_return_attachment:
            repaired = self._repair_missing_return_attachment(mol)
            if repaired is not None:
                return repaired
        if self.strict:
            raise FragmentSelfiesDecodeError("disconnected implicit anchors could not be connected")
        return _largest_component_mol(mol)

    def _group_components(self, mol: Chem.Mol) -> dict[int, set[int]]:
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        atom_to_component = {}
        for component_idx, atom_indices in enumerate(fragments):
            for atom_idx in atom_indices:
                atom_to_component[atom_idx] = component_idx

        group_components: dict[int, set[int]] = defaultdict(set)
        for placed in self.placed:
            for atom_idx in placed.atom_map.values():
                component_idx = atom_to_component.get(atom_idx)
                if component_idx is not None:
                    group_components[placed.group_id].add(component_idx)
        return group_components

    def _groups_are_connected(self, mol: Chem.Mol) -> bool:
        return all(
            len(component_indices) <= 1
            for component_indices in self._group_components(mol).values()
        )

    def _repair_missing_return_attachment(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Add one synthetic return attachment for linker generations missing it."""
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragments) <= 1:
            return mol

        atom_to_component = {}
        for component_idx, atom_indices in enumerate(fragments):
            for atom_idx in atom_indices:
                atom_to_component[atom_idx] = component_idx

        placed_by_group: dict[int, list[int]] = defaultdict(list)
        for placed_idx, placed in enumerate(self.placed):
            placed_by_group[placed.group_id].append(placed_idx)

        sites = self._available_attachment_sites()
        for site in sites:
            site["component_idx"] = atom_to_component.get(site["atom_idx"])

        distance_matrix = Chem.GetDistanceMatrix(mol)
        for group_id, placed_indices in placed_by_group.items():
            if len(placed_indices) < 2:
                continue

            component_indices = set()
            for placed_idx in placed_indices:
                for atom_idx in self.placed[placed_idx].atom_map.values():
                    component_idx = atom_to_component.get(atom_idx)
                    if component_idx is not None:
                        component_indices.add(component_idx)
            if len(component_indices) != 2:
                continue

            group_sites = [
                site
                for site in sites
                if site["group_id"] == group_id and site.get("component_idx") is not None
            ]
            if len(group_sites) != 1:
                continue

            return_site = group_sites[0]
            return_atom = int(return_site["atom_idx"])
            first_placed_idx = placed_indices[0]
            second_placed_idx = placed_indices[1]
            if return_site["placed_idx"] != first_placed_idx:
                continue

            return_component = return_site["component_idx"]
            target_components = component_indices - {return_component}
            if len(target_components) != 1:
                continue
            target_component = next(iter(target_components))

            second_atoms = [
                atom_idx
                for atom_idx in self.placed[second_placed_idx].atom_map.values()
                if atom_to_component.get(atom_idx) == target_component
            ]
            if not second_atoms:
                continue

            second_atom_set = set(self.placed[second_placed_idx].atom_map.values())
            existing_attachment_parents = set()
            candidate_atoms = set()
            for placed_idx in placed_indices[1:]:
                placed = self.placed[placed_idx]
                for attachment in placed.template.attachments:
                    atom_idx = placed.atom_map.get(attachment.parent_idx)
                    if atom_idx is not None:
                        existing_attachment_parents.add(atom_idx)
                for atom_idx in placed.atom_map.values():
                    if atom_to_component.get(atom_idx) != target_component:
                        continue
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomicNum() > 1:
                        candidate_atoms.add(atom_idx)

            candidate_atoms -= existing_attachment_parents
            candidate_atoms -= second_atom_set
            candidates = []
            for atom_idx in candidate_atoms:
                if mol.GetBondBetweenAtoms(return_atom, atom_idx) is not None:
                    continue
                distance = min(float(distance_matrix[second_atom, atom_idx]) for second_atom in second_atoms)
                if distance <= 0.0 or distance > 1e6:
                    continue
                candidates.append((distance, atom_idx))

            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _distance, atom_idx in candidates:
                repaired = self._try_synthetic_return_bond(mol, return_site, atom_idx)
                if repaired is None:
                    continue
                placed_idx = int(return_site["placed_idx"])
                attachment_idx = int(return_site["attachment_idx"])
                self.placed[placed_idx].used_attachments.add(attachment_idx)
                return repaired

        return None

    def _try_synthetic_return_bond(
        self,
        mol: Chem.Mol,
        return_site: dict[str, object],
        atom_idx: int,
    ) -> Chem.Mol | None:
        return_atom = int(return_site["atom_idx"])
        if return_atom == atom_idx or mol.GetBondBetweenAtoms(return_atom, atom_idx) is not None:
            return None

        candidate = Chem.RWMol(mol)
        candidate.AddBond(return_atom, atom_idx, Chem.BondType.SINGLE)
        repaired = candidate.GetMol()
        try:
            Chem.SanitizeMol(repaired)
            Chem.AssignStereochemistry(repaired, force=True)
        except Exception:
            return None
        if not self._groups_are_connected(repaired):
            return None
        return repaired

    def _link_components_through_unused_attachments(self, mol: Chem.Mol) -> Chem.Mol | None:
        fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragments) <= 1:
            return mol

        atom_to_component = {}
        for component_idx, atom_indices in enumerate(fragments):
            for atom_idx in atom_indices:
                atom_to_component[atom_idx] = component_idx

        sites = self._available_attachment_sites()
        for site in sites:
            site["component_idx"] = atom_to_component.get(site["atom_idx"])
        sites = [site for site in sites if site["component_idx"] is not None]

        group_components: dict[int, set[int]] = defaultdict(set)
        for placed in self.placed:
            for atom_idx in placed.atom_map.values():
                component_idx = atom_to_component.get(atom_idx)
                if component_idx is not None:
                    group_components[placed.group_id].add(component_idx)

        initial_parent = list(range(len(fragments)))
        max_attempts = 20000
        attempts = 0

        def find(parent: list[int], idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def disconnected_group(parent: list[int]) -> int | None:
            for group_id, component_indices in group_components.items():
                if len({find(parent, idx) for idx in component_indices}) > 1:
                    return group_id
            return None

        def build_linked_mol(bonds: list[tuple[int, int, Chem.BondType]]) -> Chem.Mol | None:
            candidate = Chem.RWMol(mol)
            for left_atom, right_atom, bond_type in bonds:
                if candidate.GetBondBetweenAtoms(left_atom, right_atom) is None:
                    candidate.AddBond(left_atom, right_atom, bond_type)
            linked = candidate.GetMol()
            try:
                Chem.SanitizeMol(linked)
                Chem.AssignStereochemistry(linked, force=True)
            except Exception:
                return None
            return linked

        def search(
            parent: list[int],
            used_sites: set[tuple[int, int]],
            bonds: list[tuple[int, int, Chem.BondType]],
        ) -> tuple[Chem.Mol, set[tuple[int, int]]] | None:
            nonlocal attempts
            attempts += 1
            if attempts > max_attempts:
                return None

            group_id = disconnected_group(parent)
            if group_id is None:
                linked = build_linked_mol(bonds)
                if linked is None:
                    return None
                return linked, used_sites

            group_sites = [site for site in sites if site["group_id"] == group_id]
            pairs = []
            for left_idx, left in enumerate(group_sites):
                left_key = (left["placed_idx"], left["attachment_idx"])
                if left_key in used_sites:
                    continue
                left_root = find(parent, left["component_idx"])
                for right in group_sites[left_idx + 1 :]:
                    right_key = (right["placed_idx"], right["attachment_idx"])
                    if right_key in used_sites:
                        continue
                    right_root = find(parent, right["component_idx"])
                    if left_root == right_root:
                        continue
                    left_atom = left["atom_idx"]
                    right_atom = right["atom_idx"]
                    if mol.GetBondBetweenAtoms(left_atom, right_atom) is not None:
                        continue
                    pairs.append((left_key, right_key, left, right))

            pairs.sort(key=lambda item: (item[0], item[1]))
            for left_key, right_key, left, right in pairs:
                next_parent = list(parent)
                left_root = find(next_parent, left["component_idx"])
                right_root = find(next_parent, right["component_idx"])
                next_parent[right_root] = left_root
                bond_type = (
                    left["bond_type"]
                    if left["bond_type"] == right["bond_type"]
                    else Chem.BondType.SINGLE
                )
                result = search(
                    next_parent,
                    used_sites | {left_key, right_key},
                    bonds + [(left["atom_idx"], right["atom_idx"], bond_type)],
                )
                if result is not None:
                    return result
            return None

        result = search(initial_parent, set(), [])
        if result is None:
            return None

        linked, used_sites = result
        for placed_idx, attachment_idx in used_sites:
            self.placed[placed_idx].used_attachments.add(attachment_idx)
        return linked

    def _best_effort_mol(self) -> Chem.Mol:
        candidate = Chem.RWMol()
        for placed in self.placed:
            atom_map = {}
            for atom_idx in placed.template.core_atom_indices:
                atom = placed.template.mol.GetAtomWithIdx(atom_idx)
                atom_map[atom_idx] = candidate.AddAtom(_copy_atom(atom))
            for bond in placed.template.mol.GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in atom_map and end in atom_map:
                    candidate.AddBond(atom_map[begin], atom_map[end], bond.GetBondType())
        mol = candidate.GetMol()
        try:
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, force=True)
        except Exception:
            return _fallback_valid_mol()
        if mol.GetNumAtoms() == 0:
            return _fallback_valid_mol()
        return self._linked_or_largest_component_mol(mol)


def _fallback_valid_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("C")
    if mol is None:
        raise FragmentSelfiesDecodeError("internal fallback molecule is invalid")
    return mol


def _largest_component_mol(mol: Chem.Mol) -> Chem.Mol:
    try:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return _fallback_valid_mol()
    if not fragments:
        return _fallback_valid_mol()
    largest = max(
        fragments,
        key=lambda item: (
            sum(1 for atom in item.GetAtoms() if atom.GetAtomicNum() > 1),
            item.GetNumAtoms(),
            Chem.MolToSmiles(item, canonical=True, isomericSmiles=False),
        ),
    )
    try:
        Chem.SanitizeMol(largest)
        Chem.AssignStereochemistry(largest, force=True)
    except Exception:
        return _fallback_valid_mol()
    return largest


def encode(
    mol_or_smiles: Chem.Mol | str,
    vocabulary: FragmentVocabulary | None = None,
    *,
    fallback_selfies: bool = True,
    canonical: bool = False,
    randomized: bool = False,
    seed: int | None = None,
    implicit_probability: float = DEFAULT_IMPLICIT_PROBABILITY,
    max_implicit_cuts: int = DEFAULT_MAX_IMPLICIT_CUTS,
    fragment_style: str = FRAGMENT_STYLE_AUTO,
) -> str:
    return FragmentSelfiesCodec(vocabulary).encode(
        mol_or_smiles,
        fallback_selfies=fallback_selfies,
        canonical=canonical,
        randomized=randomized,
        seed=seed,
        implicit_probability=implicit_probability,
        max_implicit_cuts=max_implicit_cuts,
        fragment_style=fragment_style,
    )


def encode_fragment(
    fragment_mol_or_smiles: Chem.Mol | str,
    vocabulary: FragmentVocabulary | None = None,
    *,
    canonical: bool = False,
    randomized: bool = False,
    seed: int | None = None,
    include_fragment_token: bool = True,
) -> str:
    return FragmentSelfiesCodec(vocabulary).encode_fragment(
        fragment_mol_or_smiles,
        canonical=canonical,
        randomized=randomized,
        seed=seed,
        include_fragment_token=include_fragment_token,
    )


def reserialize(
    fragment_selfies: str,
    vocabulary: FragmentVocabulary | None = None,
    *,
    fallback_selfies: bool = True,
    canonical: bool = True,
    randomized: bool = False,
    seed: int | None = None,
    implicit_probability: float = DEFAULT_IMPLICIT_PROBABILITY,
    max_implicit_cuts: int = DEFAULT_MAX_IMPLICIT_CUTS,
    fragment_style: str = FRAGMENT_STYLE_EXPLICIT,
    strict: bool = False,
    repair_missing_return_attachment: bool = False,
) -> str:
    return FragmentSelfiesCodec(vocabulary).reserialize(
        fragment_selfies,
        fallback_selfies=fallback_selfies,
        canonical=canonical,
        randomized=randomized,
        seed=seed,
        implicit_probability=implicit_probability,
        max_implicit_cuts=max_implicit_cuts,
        fragment_style=fragment_style,
        strict=strict,
        repair_missing_return_attachment=repair_missing_return_attachment,
    )


def decode(
    fragment_selfies: str,
    vocabulary: FragmentVocabulary | None = None,
    *,
    strict: bool = False,
    repair_missing_return_attachment: bool = False,
) -> Chem.Mol:
    return FragmentSelfiesCodec(vocabulary).decode(
        fragment_selfies,
        strict=strict,
        repair_missing_return_attachment=repair_missing_return_attachment,
    )


def decode_fragment(
    fragment_selfies: str,
    vocabulary: FragmentVocabulary | None = None,
) -> Chem.Mol:
    return FragmentSelfiesCodec(vocabulary).decode_fragment(fragment_selfies)
