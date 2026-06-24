# Fragment-SELFIES Documentation

Last updated: 2026-06-25

## Purpose

Fragment-SELFIES is a compact BRICS-fragment molecular string representation. It
is intended for molecular pretraining, fine-tuning, and generation data.
The representation records the BRICS fragment tree explicitly and encodes each
fragment body using SELFIES-compatible tokens plus non-atomic Dummy attachment
placeholders.

The current implementation is vocabulary-free for normal encode/decode. It does
not emit corpus-specific fragment-ID tokens such as `[F:F000001]`. A string is
composed of a small set of Fragment-SELFIES structural tokens plus
SELFIES-compatible fragment-body tokens.

Examples in this document use standalone SMILES strings. For corpus-scale use,
pass your own `.smi` file paths, such as `/path/to/molecules.smi`, to the CLI.
The default representation is non-isomeric, so it intentionally does not preserve
stereochemistry.

## Design Goals

- Preserve chemically meaningful BRICS fragment structure.
- Avoid huge fragment-ID vocabularies and rare fragment ID tokens.
- Represent fragment internals with SELFIES-compatible tokens.
- Use a small fixed set of structural tokens for tree traversal.
- Support deterministic default encoding, canonical encoding, and seeded randomized encoding.
- Decode compact strings back to RDKit molecules.
- Keep the representation easy for tokenizers to learn as bracketed tokens.
- Avoid global mutable grammars and hand-written SMARTS matching for every fragment token.

## High-Level Structure

A Fragment-SELFIES string has two layers:

| Layer | Tokens | Responsibility |
| --- | --- | --- |
| Fragment-SELFIES structure | `[Frag]`, `[Frag@N]`, `[Attach:N]`, `[pop]`, `[.]` | Describes how BRICS fragments are arranged and connected. |
| SELFIES-like fragment body | `[C]`, `[=C]`, `[Branch1]`, `[Ring1]`, `[Dummy]`, etc. | Describes the atoms, bonds, branches, rings, and attachment placeholders inside each fragment. |

In one sentence: current compact Fragment-SELFIES is a strict bracketed-token
string where `[Frag]` starts root fragments, `[Frag@N]` starts child fragments,
fragment-body tokens describe each BRICS fragment with SELFIES-compatible syntax,
`[Attach:N][Frag@M]` serializes explicit BRICS tree edges, and adjacent root
`[Frag]` blocks represent implicit anchors.

Informal grammar:

```text
fragment_selfies := item+
item             := root_fragment | component_separator | selfies_fallback
root_fragment    := "[Frag]" fragment_body edge*
child_fragment   := "[Frag@" N "]" fragment_body edge*
edge             := "[Attach:" N "]" child_fragment "[pop]"
fragment_body    := SELFIES-compatible tokens excluding structural tokens
component_separator := "[.]"
selfies_fallback := "[SELFIES]" official_SELFIES_tokens "[ENDSELFIES]"
```

This grammar is intentionally tree-shaped for explicit BRICS edges. Implicit
anchors are represented by placing multiple `root_fragment` items next to each
other without `[.]` and without an explicit `[Attach:N][Frag@M]` edge.

Example:

```text
[Frag][C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C][Attach:0][Frag@1][O][=C][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][O][Dummy][pop][pop]
```

The structural tokens split this into three fragment bodies:

```text
root fragment body:
[C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C]

child fragment body:
[O][=C][Branch1][C][Dummy][Dummy]

grandchild fragment body:
[C][O][Dummy]
```

## Token Grammar

All tokens are bracketed. Non-whitespace text outside bracketed tokens is
invalid.

Structural Fragment-SELFIES tokens:

| Token | Meaning |
| --- | --- |
| `[Frag]` | Start a root fragment. Adjacent root fragments are disconnected anchors unless an explicit attachment edge is emitted. |
| `[Frag@N]` | Start a child fragment and select child attachment point `N` for the incoming connection. |
| `[Attach:N]` | Select attachment point `N` on the current parent fragment. |
| `[pop]` | Return traversal state from the current child fragment to its parent. |
| `[.]` | Legacy disconnected molecule component marker. |
| `[SELFIES]...[ENDSELFIES]` | Store a whole-molecule official SELFIES fallback block. |

Fragment body tokens are SELFIES-compatible tokens. Examples:

| Token | Meaning in this context |
| --- | --- |
| `[C]` | SELFIES carbon token. |
| `[O]` | SELFIES oxygen token. |
| `[N]` | SELFIES nitrogen token. |
| `[=C]` | SELFIES carbon token with double-bond information. |
| `[Branch1]` | SELFIES branch-control token. |
| `[Ring1]` | SELFIES ring-closure token. |
| `[=Branch1]` | SELFIES parameter token often consumed by ring/branch syntax and carrying bond-order information. |
| `[Dummy]` | Non-atomic attachment placeholder in Fragment-SELFIES compact mode. |

Important distinction: `[Frag]`, `[Frag@N]`, `[Attach:N]`, `[pop]`, and `[.]` are
Fragment-SELFIES structural tokens. Tokens such as `[Ring1]` and `[=Branch1]`
are standard SELFIES tokens inside a fragment body, while `[Dummy]` marks an
attachment placeholder.

## Explicit And Implicit Anchor Semantics

Fragment-SELFIES supports implicit-anchor prompts without introducing tokens that a
model pretrained on the compact corpus has not seen. Adjacent root
fragments represent disconnected anchors:

```text
[Frag]anchor_A[Frag]anchor_B
```

There are no special implicit tokens. An implicit Fragment-SELFIES string looks
like ordinary compact Fragment-SELFIES whose cut edge has been emitted as a new
adjacent root instead of as an explicit child edge.

Explicit tree edges still use `[Attach:N]` followed by `[Frag@M]`:

```text
[Frag]fragment_A[Attach:0][Frag@0]fragment_B
```

That explicit form means the edge is already present in the serialized BRICS
tree. It is not an implicit-anchor prompt.

Likewise, `[.]` is not the preferred implicit-anchor separator:

```text
[Frag]anchor_A[.][Frag]anchor_B
```

That form means legacy disconnected components, not adjacent anchors in the same
repair group.

Concrete two-anchor shape:

```text
[Frag][C][=C][C][=C][C][=C][Ring1][=Branch1][Dummy][Frag][O][=C][Dummy]
```

Interpretation:

```text
anchor A:
[Frag][C][=C][C][=C][C][=C][Ring1][=Branch1][Dummy]

anchor B:
[Frag][O][=C][Dummy]
```

Both anchors contain `[Dummy]` placeholders. Those are the open attachment sites
the decoder can use to reconnect the anchors.

A randomized training-time implicit-augmented string can therefore begin like this:

```text
[Frag][O][CH0][=Branch1][C][=O][Dummy][Frag][C][=C]...
```

The key detail is that there is no `[Attach:N][Frag@M]` between the first two
`[Frag]` roots, so the selected BRICS edge has been represented as implicit-anchor
adjacency.

This distinction makes implicit prompting possible with the compact vocabulary:
the prompt can provide two or more anchor fragments as adjacent `[Frag]...`
blocks, and generation can continue from the last anchor. During decode,
Fragment-SELFIES attempts to connect adjacent root-anchor components through
unused `[Dummy...]` attachment points. If a valid connected molecule can be
sanitized, all anchor atoms are preserved. If linking is impossible, strict
decode raises and non-strict decode falls back to the largest valid component.

Linker-design generations sometimes consume every generated-side `[Dummy]`
placeholder before returning to the first anchor. In that case, the decoder can
optionally synthesize one missing return attachment with
`repair_missing_return_attachment=True` or the CLI flag
`--repair-missing-return-attachment`. The repair is conservative: it only applies
when the first anchor has exactly one unused attachment, the second anchor is in
the generated linker component, and adding a single bond from the first anchor to
the farthest topologically suitable generated atom passes RDKit sanitization.

Mental model:

| Syntax | Meaning |
| --- | --- |
| `[Frag]A[Frag]B` | `A` and `B` are implicit anchors; connect through unused Dummy sites if possible. |
| `[Frag]A[Attach:0][Frag@1]B[pop]` | `A` is explicitly connected to `B` in the serialized BRICS tree. |
| `[Frag]A[.][Frag]B` | `A` and `B` are legacy disconnected components. |

`[.]` remains accepted as a legacy disconnected-component marker, but pretrained
tokenizers may not contain it. Prefer adjacent root `[Frag]` blocks for implicit
prompts intended for pretrained models.

## Why `[Dummy]` Is Used As The Attachment Placeholder

BRICS fragmentation creates dummy atoms (`*`) at cut sites. A fragment may look
like this as a SMILES fragment:

```text
*c1ccccc1
```

The `*` means the fragment connects to another fragment at that site. Standard
SELFIES does not support RDKit dummy atoms directly, so Fragment-SELFIES uses an
internal temporary sentinel only while calling the SELFIES encoder and decoder.
That sentinel is never serialized. The public Fragment-SELFIES string uses
non-atomic `[Dummy...]` tokens instead.

Example transformation:

```text
*c1ccccc1
```

becomes:

```text
[Dummy]c1ccccc1
```

which Fragment-SELFIES serializes with placeholder tokens such as:

```text
[Dummy]
[=Dummy]
[#Dummy]
```

Reasons for using Dummy tokens:

- Real elements always remain real chemistry in the serialized representation.
- Dummy tokens can preserve local bond-order information needed at cut sites.
- The decoder can identify Dummy placeholders and omit them from the final molecule.
- Real xenon-containing molecules are no longer ambiguous with attachment sites.

In compact Fragment-SELFIES, `[Dummy]` means attachment placeholder, not a real
atom. Real xenon and other supported elements remain ordinary fragment-body
tokens.

## Rings And Branches In Fragment Bodies

Tokens such as `[Ring1]`, `[Branch1]`, `[=Branch1]`, and `[#Branch1]` are
standard SELFIES tokens. Fragment-SELFIES does not reinterpret them as structural
tree tokens.

For example, in a fragment body:

```text
[C][C][=C][C][=N][C][=C][Ring1][=Branch1]
```

`[Ring1]` instructs the SELFIES decoder to create a ring closure back to a
previous atom. The `1` indicates that one following SELFIES token is used as the
ring parameter. In this example, `[=Branch1]` is consumed as that parameter and
also carries bond-order information. This is roughly analogous to a ring digit in
SMILES, such as the `1` in `c1ccccc1`, but encoded as robust SELFIES tokens.

## Encoding Modes

The codec exposes three practical modes.

| Mode | API flags | Behavior |
| --- | --- | --- |
| Default deterministic | no flags | Preserves input order where possible and uses deterministic traversal. |
| Canonical | `canonical=True` | Canonicalizes the input molecule before encoding and emits a stable deterministic string. |
| Randomized | `randomized=True, seed=N` | Randomizes atom order, fragment traversal, and probabilistic implicit edge cuts for data augmentation. |

`canonical=True` and `randomized=True` are mutually exclusive.

Python API:

```python
from fragment_selfies import FragmentSelfiesCodec

codec = FragmentSelfiesCodec()

default_encoded = codec.encode("COC(=O)c1c(N)oc(C)c1C")
canonical_encoded = codec.encode("COC(=O)c1c(N)oc(C)c1C", canonical=True)
randomized_encoded = codec.encode("COC(=O)c1c(N)oc(C)c1C", randomized=True, seed=0)
implicit_augmented = codec.encode(
    "COC(=O)c1c(N)oc(C)c1C",
    randomized=True,
    seed=0,
    implicit_probability=1.0,
    max_implicit_cuts=2,
)
explicit_encoded = codec.encode(
    "COC(=O)c1c(N)oc(C)c1C",
    randomized=True,
    fragment_style="explicit",
)
implicit_encoded = codec.encode(
    "COC(=O)c1c(N)oc(C)c1C",
    canonical=True,
    fragment_style="implicit",
)
```

`implicit_probability` is used only during randomized `fragment_style="auto"`
encoding. It defaults to `0.15`, meaning each connected component has a 15%
chance of cutting implicit BRICS edges into adjacent root anchors. The number of
cuts is randomly selected from `1..max_implicit_cuts`, capped by the number of
available BRICS edges. The default `max_implicit_cuts=1` preserves two-root
implicit samples. Set `max_implicit_cuts=2` or higher to generate multi-root
implicit samples for multi-fragment design training. Set `fragment_style="explicit"`
to force explicit `[Attach:N][Frag@M]` edges, or `fragment_style="implicit"` to
force up to `max_implicit_cuts` adjacent-root implicit edges per connected
component when BRICS edges exist.

CLI:

```bash
conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --canonical

conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --randomized \
  --seed 0 \
  --implicit-probability 0.15 \
  --max-implicit-cuts 2

conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --canonical \
  --style implicit \
  --max-implicit-cuts 2

conda run -n fragment_selfies fragment-selfies reserialize \
  --fragment-selfies '[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Frag][C][OH0][Dummy]' \
  --style explicit \
  --canonical
```

Decode and reserialize commands use non-strict recovery mode by default for
generated strings. Add `--strict` when the input should be treated as a strict
Fragment-SELFIES validation target.

For linker-design outputs with a missing return attachment, add
`--repair-missing-return-attachment` to `decode` or `reserialize`.

## Encoding Algorithm

The encoder performs these steps:

1. Parse the input SMILES or clone the input RDKit molecule.
2. If `canonical=True`, convert the molecule to canonical non-isomeric SMILES and parse it again.
3. If `randomized=True`, shuffle atom order with the selected RNG.
4. Find BRICS bonds with RDKit.
5. Fragment the molecule on those BRICS bonds.
6. Label cut dummy atoms with molecule-local connection IDs so the original fragment adjacency can be reconstructed.
7. Replace dummy atoms in each fragment with temporary internal sentinels for SELFIES encoding.
8. Convert each placeholder-containing fragment to SELFIES tokens and serialize sentinel tokens as `[Dummy...]`.
9. Determine attachment order in each fragment by canonical ranking of placeholder atoms.
10. Match connection IDs across fragments to build the BRICS fragment tree.
11. Choose root order and child traversal order.
12. During randomized auto-style encoding, optionally select one BRICS edge per connected component for implicit-anchor augmentation.
13. Emit `[Frag]` or `[Frag@N]`, fragment body tokens, `[Attach:N]`, `[pop]`, and `[.]` as needed. Selected implicit edges are emitted as new adjacent `[Frag]` roots without `[Attach:N]`.
14. If fragment SELFIES encoding fails and fallback is enabled, return `[SELFIES]... [ENDSELFIES]` around a whole-molecule official SELFIES block.

## Decoding Algorithm

The decoder reverses the two-layer representation.

1. Split the input string into bracketed tokens.
2. When `[SELFIES]` is found, decode the enclosed whole-molecule SELFIES block and add it as a disconnected molecule.
3. When `[Frag]` or `[Frag@N]` is found, collect following tokens until the next structural token.
4. Temporarily map `[Dummy...]` tokens back to the internal sentinel and decode the collected fragment-body tokens with the standard SELFIES decoder.
5. Parse the decoded fragment SMILES with RDKit.
6. Identify sentinel atoms as attachment placeholders.
7. Copy only non-placeholder core atoms into an editable RDKit molecule.
8. Record which core atom each placeholder was attached to and the original placeholder bond type.
9. When `[Attach:N]` is read, select attachment `N` on the current parent fragment.
10. When a child `[Frag@M]` is placed, select attachment `M` on that child fragment.
11. Add a bond between the selected parent core atom and the selected child core atom.
12. When `[pop]` is read, return traversal state to the parent fragment.
13. If multiple adjacent root-anchor components remain in the same group, try to
    link them through unused attachment points.
14. If opt-in missing-return repair is enabled and one anchor group still lacks
    a generated-side return site, test farthest suitable synthetic return bonds.
15. After all tokens are consumed, sanitize the assembled RDKit molecule.
16. If anchor linking is impossible, strict decoding raises and non-strict
    decoding, the default, falls back to the largest valid component.

Dummy placeholders never survive into the decoded molecule. They only mark where
BRICS fragments reconnect, while real elements remain ordinary chemistry.

## Detailed Token-By-Token Example

This example uses a representative small-molecule SMILES string.

Input SMILES:

```text
COC(=O)c1c(N)oc(C)c1C
```

Canonical Fragment-SELFIES:

```text
[Frag][C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C][Attach:0][Frag@1][O][=C][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][O][Dummy][pop][pop]
```

Token-by-token interpretation:

| # | Token | Interpretation |
| ---: | --- | --- |
| 1 | `[Frag]` | Start the root BRICS fragment. |
| 2 | `[C]` | SELFIES token in the root fragment body. |
| 3 | `[C]` | SELFIES carbon token. |
| 4 | `[O]` | SELFIES oxygen token. |
| 5 | `[C]` | SELFIES carbon token. |
| 6 | `[Branch1]` | SELFIES branch-control token inside the fragment. |
| 7 | `[C]` | Branch/body atom token. |
| 8 | `[N]` | Nitrogen token. |
| 9 | `[=C]` | Double-bond carbon token. |
| 10 | `[Branch1]` | Another SELFIES branch-control token. |
| 11 | `[C]` | Branch/body atom token. |
| 12 | `[Dummy]` | Attachment placeholder inside the root fragment. |
| 13 | `[C]` | SELFIES carbon token. |
| 14 | `[=Ring1]` | SELFIES ring-closure token inside the fragment body. |
| 15 | `[#Branch1]` | SELFIES branch/ring parameter token. |
| 16 | `[C]` | Final token of the root fragment body. |
| 17 | `[Attach:0]` | Select attachment point `0` on the current root fragment. |
| 18 | `[Frag@1]` | Start a child fragment and use child attachment point `1` for the incoming connection. |
| 19 | `[O]` | SELFIES token in the child fragment body. |
| 20 | `[=C]` | Carbonyl-like double-bond carbon token. |
| 21 | `[Branch1]` | SELFIES branch-control token. |
| 22 | `[C]` | Branch/body atom token. |
| 23 | `[Dummy]` | Attachment placeholder in the child fragment. |
| 24 | `[Dummy]` | Another attachment placeholder in the child fragment. |
| 25 | `[Attach:0]` | Select attachment point `0` on the current child fragment. |
| 26 | `[Frag@0]` | Start a grandchild fragment and use child attachment `0`. |
| 27 | `[C]` | SELFIES token in the grandchild fragment body. |
| 28 | `[O]` | Oxygen token. |
| 29 | `[Dummy]` | Attachment placeholder in the grandchild fragment. |
| 30 | `[pop]` | Return from the grandchild fragment to its parent. |
| 31 | `[pop]` | Return from the child fragment to the root. |

The root, child, and grandchild fragments are connected by the `[Attach:N]` and
`[Frag@N]` selections. The final decoded canonical non-isomeric SMILES is:

```text
COC(=O)c1c(N)oc(C)c1C
```

## Detailed Decode Derivation Example

This example uses a representative multi-fragment molecule.

Input SMILES:

```text
Cc1ccnc2c1[nH]c(=N)n2-c1ccc(Br)c(F)c1
```

Canonical Fragment-SELFIES:

```text
[Frag][C][C][=C][C][=N][C][=C][Ring1][=Branch1][NH1][C][=Branch1][C][=N][N][Ring1][=Branch1][Dummy][Attach:0][Frag@0][F][C][=C][C][Branch1][C][Dummy][=C][C][=C][Ring1][#Branch1][Br][pop]
```

Step 1: split the string into a root fragment, parent attachment selection, child
fragment, and pop:

```text
[Frag]
[C][C][=C][C][=N][C][=C][Ring1][=Branch1][NH1][C][=Branch1][C][=N][N][Ring1][=Branch1][Dummy]
[Attach:0]
[Frag@0]
[F][C][=C][C][Branch1][C][Dummy][=C][C][=C][Ring1][#Branch1][Br]
[pop]
```

Step 2: map `[Dummy]` to the internal sentinel and decode the root fragment body:

```text
[C][C][=C][C][=N][C][=C][Ring1][=Branch1][NH1][C][=Branch1][C][=N][N][Ring1][=Branch1][Dummy]
```

Conceptual root fragment SMILES:

```text
CC1=CC=NC2=C1[NH1]C(=N)N2[Dummy]
```

Conceptual canonical form:

```text
Cc1ccnc2c1[nH]c(=N)n2[Dummy]
```

The Dummy atom marks root attachment point `0`. The Dummy atom itself is not
copied to the final molecule.

Step 3: read `[Attach:0]`.

This selects root attachment `0` as the connection site for the next child.

Step 4: decode the child fragment body after `[Frag@0]`:

```text
[F][C][=C][C][Branch1][C][Dummy][=C][C][=C][Ring1][#Branch1][Br]
```

Temporary child fragment SMILES:

```text
FC1=CC([Dummy])=CC=C1Br
```

Conceptual canonical form:

```text
Fc1cc([Dummy])ccc1Br
```

The `@0` in `[Frag@0]` selects child attachment `0`.

Step 5: connect the selected parent and child attachment atoms.

Conceptually:

```text
Cc1ccnc2c1[nH]c(=N)n2[Dummy]
Fc1cc([Dummy])ccc1Br
```

becomes:

```text
Cc1ccnc2c1[nH]c(=N)n2-c1ccc(Br)c(F)c1
```

Step 6: read `[pop]`.

The traversal state returns from the child to the root. No more tokens remain, so
the decoder sanitizes the RDKit molecule.

Final decoded canonical non-isomeric SMILES:

```text
Cc1ccnc2c1[nH]c(=N)n2-c1ccc(Br)c(F)c1
```

## Implicit Anchor Decode Derivation Example

This example shows a string with adjacent root `[Frag]` blocks. The second root
fragment is an implicit anchor rather than a legacy disconnected component,
because it is not preceded by `[.]`.

Input Fragment-SELFIES:

```text
[Frag][CH0][Branch1][C][Dummy][Branch1][C][Dummy][C][CH1][Branch1][C][Dummy][C][C][C][Ring1][=Branch2][Attach:1][Frag@0][C][NH1][Dummy][pop][Attach:0][Frag@0][OH0][Branch1][C][Dummy][Dummy][pop][Attach:2][Frag@0][O][=CH0][Branch1][C][O][Dummy][pop][Frag][CH1][Branch1][C][Dummy][C][C][Ring1][Ring2][Attach:0][Frag@1][CH2][Branch1][C][Dummy][Dummy][pop]
```

Decoded canonical non-isomeric SMILES:

```text
CNC1(C(=O)O)CCCC(OCC2CC2)C1
```

The string contains six fragment bodies:

| Fragment | Marker | Internal fragment SMILES | Role |
| --- | --- | --- | --- |
| 1 | `[Frag]` | `[Lv][CH]1CCC[C]([Lv])([Lv])C1` | Main cyclohexane-like root with three attachment sites. |
| 2 | `[Frag@0]` | `C[NH][Lv]` | Methylamino substituent. |
| 3 | `[Frag@0]` | `[Lv][O][Lv]` | Ether oxygen bridge. |
| 4 | `[Frag@0]` | `O=[C](O)[Lv]` | Carboxylic acid substituent. |
| 5 | `[Frag]` | `[Lv][CH]1CC1` | Cyclopropyl adjacent root anchor. |
| 6 | `[Frag@1]` | `[Lv][CH2][Lv]` | Methylene bridge attached to the cyclopropyl anchor. |

`[Lv]` is the decoder's temporary internal attachment sentinel after serialized
`[Dummy]` tokens are deserialized. These placeholder atoms are not copied into
the final molecule; they only identify connection sites.

Step 1: parse the first root fragment.

```text
[Frag][CH0][Branch1][C][Dummy][Branch1][C][Dummy][C][CH1][Branch1][C][Dummy][C][C][C][Ring1][=Branch2]
```

This decodes to a main ring fragment with three attachment sites:

```text
[Lv][CH]1CCC[C]([Lv])([Lv])C1
```

Step 2: attach fragment 2 through an explicit edge.

```text
[Attach:1][Frag@0][C][NH1][Dummy][pop]
```

`[Attach:1]` selects attachment `1` on the main ring. `[Frag@0]` selects
attachment `0` on the child fragment `C[NH][Lv]`. The decoder connects those two
attachment parent atoms, producing the methylamino substituent represented in the
final SMILES as `CNC1...`.

Step 3: attach fragment 3 through another explicit edge.

```text
[Attach:0][Frag@0][OH0][Branch1][C][Dummy][Dummy][pop]
```

The child body decodes to:

```text
[Lv][O][Lv]
```

One oxygen attachment connects to attachment `0` on the main ring. The other
oxygen attachment remains unused at this point and will be used later for
implicit-anchor repair.

Step 4: attach fragment 4 through another explicit edge.

```text
[Attach:2][Frag@0][O][=CH0][Branch1][C][O][Dummy][pop]
```

The child body decodes to:

```text
O=[C](O)[Lv]
```

Connecting its attachment to main-ring attachment `2` forms the carboxylic acid
substituent `C(=O)O` on the ring.

Step 5: parse the adjacent root implicit anchor.

```text
[Frag][CH1][Branch1][C][Dummy][C][C][Ring1][Ring2]
```

This second `[Frag]` starts a new root component adjacent to the first root
component. Because there is no `[.]`, it remains in the same implicit-repair group.
The fragment body decodes to a cyclopropyl anchor:

```text
[Lv][CH]1CC1
```

Step 6: attach fragment 6 to the cyclopropyl anchor.

```text
[Attach:0][Frag@1][CH2][Branch1][C][Dummy][Dummy][pop]
```

The child body decodes to:

```text
[Lv][CH2][Lv]
```

Attachment `1` on this methylene fragment connects to attachment `0` on the
cyclopropyl anchor. Its other attachment remains unused.

Step 7: repair the adjacent root anchors.

After explicit edges have been processed, the decoder has two components in the
same implicit group:

```text
component A: main ring + methylamino + carboxylic acid + ring-O-[open]
component B: [open]-CH2-cyclopropyl
```

The unused attachment on the ether oxygen from fragment 3 and the unused
attachment on the methylene from fragment 6 are compatible open sites. The
decoder adds a single bond between those parent atoms:

```text
ring-O-CH2-cyclopropyl
```

After this implicit-anchor repair, RDKit sanitization succeeds and the complete
connected molecule is:

```text
CNC1(C(=O)O)CCCC(OCC2CC2)C1
```

The important syntax transition is the adjacent root boundary:

```text
...[pop][Frag][CH1]...
```

That second `[Frag]` starts an adjacent implicit anchor. If the string had used
`[.][Frag]` instead, the two root components would be legacy disconnected
components rather than anchors intended for implicit repair.

## Canonical Encoding And Randomized Encoding Examples

The following examples use representative SMILES strings. For each molecule,
canonical and randomized encodings decode back to the same
canonical non-isomeric SMILES. Randomized strings differ because atom order and
fragment traversal are randomized.

### Example 1: line 1

SMILES:

```text
COC(=O)c1c(N)oc(C)c1C
```

Canonical encoding, 31 tokens, 3 fragments:

```text
[Frag][C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C][Attach:0][Frag@1][O][=C][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][O][Dummy][pop][pop]
```

Randomized encoding with seed `0`, 37 tokens, 3 fragments:

```text
[Frag][C][C][=C][Branch1][C][C][C][Branch1][C][Dummy][=C][Branch1][C][N][O][Ring1][Branch2][Attach:0][Frag@1][C][=Branch1][C][=O][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][O][Branch1][C][C][Dummy][pop][pop]
```

Both decode to:

```text
COC(=O)c1c(N)oc(C)c1C
```

### Example 2: line 2

SMILES:

```text
Cc1ccc(N(C)C2(CN)CCCOC2)c(C)c1
```

Canonical encoding, 46 tokens, 4 fragments:

```text
[Frag][C][C][=C][C][=C][Branch1][C][Dummy][C][Branch1][C][C][=C][Ring1][Branch2][Attach:0][Frag@0][C][N][Branch1][C][Dummy][Dummy][Attach:1][Frag@0][Dummy][C][Branch1][C][Dummy][C][C][C][O][C][Ring1][#Branch1][Attach:1][Frag@0][N][C][Dummy][pop][pop][pop]
```

Randomized encoding with seed `0`, 50 tokens, 4 fragments:

```text
[Frag][C][C][=C][C][=C][Branch1][C][Dummy][C][Branch1][C][C][=C][Ring1][Branch2][Attach:0][Frag@0][N][Branch1][C][C][Branch1][C][Dummy][Dummy][Attach:1][Frag@0][C][C][C][Branch1][C][Dummy][Branch1][C][Dummy][C][O][C][Ring1][Branch2][Attach:1][Frag@0][N][C][Dummy][pop][pop][pop]
```

Both decode to:

```text
Cc1ccc(N(C)C2(CN)CCCOC2)c(C)c1
```

### Example 3: line 3

SMILES:

```text
Cc1ccnc2c1[nH]c(=N)n2-c1ccc(Br)c(F)c1
```

Canonical encoding, 35 tokens, 2 fragments:

```text
[Frag][C][C][=C][C][=N][C][=C][Ring1][=Branch1][NH1][C][=Branch1][C][=N][N][Ring1][=Branch1][Dummy][Attach:0][Frag@0][F][C][=C][C][Branch1][C][Dummy][=C][C][=C][Ring1][#Branch1][Br][pop]
```

Randomized encoding with seed `0`, 41 tokens, 2 fragments:

```text
[Frag][N][=C][N][Branch1][C][Dummy][C][=N][C][=C][C][Branch1][C][C][=C][Ring1][#Branch1][NH1][Ring1][O][Attach:0][Frag@0][C][Branch1][C][Dummy][=C][C][Branch1][C][F][=C][Branch1][C][Br][C][=C][Ring1][=Branch2][pop]
```

Both decode to:

```text
Cc1ccnc2c1[nH]c(=N)n2-c1ccc(Br)c(F)c1
```

### Example 4: line 4

SMILES:

```text
CNC1(C(=O)O)CCCC(OCC2CC2)C1
```

Canonical encoding, 52 tokens, 6 fragments:

```text
[Frag][Dummy][C][C][C][C][C][Branch1][C][Dummy][Branch1][C][Dummy][C][Ring1][Branch2][Attach:0][Frag@0][Dummy][O][Dummy][Attach:1][Frag@0][Dummy][C][Dummy][Attach:1][Frag@0][Dummy][C][C][C][Ring1][Ring1][pop][pop][pop][Attach:1][Frag@0][C][N][Dummy][pop][Attach:2][Frag@0][O][=C][Branch1][C][O][Dummy][pop]
```

Randomized encoding with seed `0`, 60 tokens, 6 fragments:

```text
[Frag][N][Branch1][C][C][Dummy][Attach:0][Frag@1][C][Branch1][C][Dummy][C][C][C][C][Branch1][C][Dummy][Branch1][C][Dummy][C][Ring1][=Branch2][Attach:2][Frag@0][O][C][=Branch1][C][=O][Dummy][pop][Attach:0][Frag@0][O][Branch1][C][Dummy][Dummy][Attach:1][Frag@0][C][Branch1][C][Dummy][Dummy][Attach:1][Frag@0][C][C][C][Ring1][Ring1][Dummy][pop][pop][pop][pop]
```

Both decode to:

```text
CNC1(C(=O)O)CCCC(OCC2CC2)C1
```

## Optional Fragment Statistics

The compact codec does not need a BRICS fragment-ID vocabulary. The historical
`build-vocab` command remains useful for fragment frequency analysis.

```bash
conda run --live-stream -n fragment_selfies fragment-selfies build-vocab \
  --input /path/to/molecules.smi \
  --output /path/to/fragment_vocab.jsonl \
  --min-count 1 \
  --workers 8 \
  --canonical \
  --progress
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--min-count N` | Drop fragments observed fewer than `N` times from the analysis artifact. |
| `--max-fragments N` | Keep only the top `N` fragments by frequency. |
| `--max-molecules N` | Analyze a bounded sample. |
| `--workers N` | Process molecules in `N` worker processes. |
| `--chunk-size N` | Send `N` molecules per worker task. |
| `--canonical` | Canonicalize input SMILES before fragmentation. |
| `--progress` | Force the progress bar on stderr. |
| `--no-progress` | Disable the progress bar. |

## BRICS Invariance Check

Before processing a large corpus, verify that randomized SMILES forms produce the
same BRICS fragment multiset:

```bash
conda run -n fragment_selfies fragment-selfies verify-brics \
  --input /path/to/molecules.smi \
  --sample-size 1000 \
  --randomizations 5 \
  --seed 29
```

Example successful result:

```json
{
  "invalid_smiles": 0,
  "is_consistent": true,
  "mismatches": [],
  "molecules_checked": 1000
}
```

## Verification

Commands:

```bash
conda run --live-stream -n fragment_selfies python -m py_compile src/fragment_selfies/*.py tests/test_fragment_selfies.py
conda run --live-stream -n fragment_selfies python -m pytest -q
```

Expected result:

```text
37 passed, 1 skipped
```

An optional corpus roundtrip test runs when `FRAGMENT_SELFIES_SAMPLE_SMI` points
to a local `.smi` file.

## Molecular Language Model Integration

A downstream model should train its tokenizer directly on compact Fragment-SELFIES
corpus files. The tokenizer only needs to learn bracketed tokens from the generated
corpus: structural Fragment-SELFIES tokens, SELFIES-compatible fragment-body
tokens, and any task-specific control tokens.

Recommended integration path:

1. Convert pretraining molecules from SMILES to compact Fragment-SELFIES.
2. Train a WordLevel tokenizer on the generated Fragment-SELFIES corpus.
3. Use `FragmentSelfiesCodec.encode()` for SMILES to Fragment-SELFIES conversion.
4. Use `FragmentSelfiesCodec.decode()` followed by `Chem.MolToSmiles()` for generated molecule recovery.

## Limitations And Edge Cases

- The grammar targets BRICS-fragment trees, not arbitrary fragment graph cycles.
- Stereochemistry is not retained in the default representation.
- `[Dummy...]` tokens are reserved as non-atomic attachment placeholders and are removed during decode.
- The fallback block encodes a whole molecule with official SELFIES and does not preserve BRICS tree structure.
- Randomized encoding may change token count because standard SELFIES can encode the same fragment through different but equivalent traversals.
- Future work can add an opt-in stereo-aware mode using `isomericSmiles=True`.
