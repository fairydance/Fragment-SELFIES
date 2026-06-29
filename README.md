# Fragment-SELFIES

Fragment-SELFIES is a robust, fragment-aware molecular language with validity-preserving decoding and explicit fragment structure. It records the BRICS
fragment tree with a small set of structural tokens, while each fragment body is
encoded as SELFIES-like tokens plus non-atomic Dummy attachment placeholders.

The current format is vocabulary-free at encode/decode time. It does not emit
large corpus-specific IDs such as `[F:F000001]`. Instead, it emits tokens like
`[Frag]`, `[Frag@0]`, `[Attach:0]`, `[pop]`, and SELFIES-compatible
fragment-body tokens such as `[C]`, `[=C]`, `[Branch1]`, `[Ring1]`, and `[Dummy]`.

Full technical details are in [`docs/fragment_selfies.md`](https://github.com/fairydance/Fragment-SELFIES/blob/main/docs/fragment_selfies.md).

![Fragment-SELFIES codec overview](https://raw.githubusercontent.com/fairydance/Fragment-SELFIES/main/images/fragment_selfies_codec.png)

## Why This Format Exists

The fragment-token approach required a huge BRICS fragment vocabulary.
For large molecular corpora, that produces large JSONL vocabulary
artifacts and many rare fragment-ID tokens. The compact representation
keeps the chemically meaningful BRICS decomposition but avoids fragment IDs by
encoding each fragment body with SELFIES-compatible tokens.

This gives molecular language-model workflows a representation with:

- Explicit BRICS fragment tree structure.
- SELFIES-compatible tokens for fragment internals.
- A small fixed set of Fragment-SELFIES structural tokens.
- Canonical and randomized encoding modes.
- Randomized implicit-anchor augmentation for design pretraining.
- Direct decoding back to RDKit molecules.
- No required fragment vocabulary file for normal encode/decode.

## Installation

Fragment-SELFIES supports Python 3.11 and newer.

```bash
python -m pip install fragment-selfies
```

RDKit is a runtime dependency. If a compatible RDKit wheel is unavailable for
your platform, install RDKit from `conda-forge` first, then install
Fragment-SELFIES with pip in the same environment.

Development install:

```bash
git clone https://github.com/fairydance/Fragment-SELFIES.git
cd Fragment-SELFIES
conda create -n fragment_selfies python=3.13
conda activate fragment_selfies
conda install -c conda-forge rdkit
python -m pip install -e ".[test]"
```

Verify the installation:

```bash
python -m pytest -q
```

Build and validate release artifacts:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Token Grammar

Fragment-SELFIES strings are strict bracketed-token strings.

In one sentence: current compact Fragment-SELFIES starts root fragments with
`[Frag]`, starts child fragments with `[Frag@N]`, stores each fragment body as
SELFIES-compatible tokens, connects explicit BRICS tree edges with
`[Attach:N][Frag@M]`, and represents implicit anchors as adjacent root
`[Frag]` blocks.

Example:

```text
[Frag][C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C][Attach:0][Frag@1][O][=C][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][O][Dummy][pop][pop]
```

Structural Fragment-SELFIES tokens:

| Token | Meaning |
| --- | --- |
| `[Frag]` | Start a root BRICS fragment. Adjacent root fragments are disconnected anchors unless an explicit attachment edge is emitted. |
| `[Frag@N]` | Start a child fragment and connect through child attachment `N`. |
| `[Attach:N]` | Select parent attachment `N` on the current fragment. |
| `[pop]` | Return traversal state to the parent fragment. |
| `[.]` | Legacy disconnected molecule component marker. |
| `[SELFIES]...[ENDSELFIES]` | Optional whole-molecule SELFIES fallback block. |

Everything between a fragment marker and the next structural token is the
SELFIES-compatible body for that fragment.

The `[Dummy...]` tokens are non-atomic attachment placeholders. They are used
because the standard SELFIES encoder cannot encode RDKit dummy atoms (`*`)
directly, and they avoid colliding with real elements such as xenon.

## Explicit And Implicit Anchor Semantics

Fragment-SELFIES uses only tokens that can be learned from the compact molecular
corpus. This lets a pretrained tokenizer accept implicit-anchor prompts without
adding new vocabulary.

There are no special implicit tokens. An implicit string is normal compact
Fragment-SELFIES whose selected cut edge appears as adjacent roots:

Use adjacent root fragments as disconnected anchors:

```text
[Frag]anchor_A[Frag]anchor_B
```

A concrete two-anchor shape is:

```text
[Frag][C][=C][C][=C][C][=C][Ring1][=Branch1][Dummy][Frag][O][=C][Dummy]
```

Both anchors contain `[Dummy]` placeholders, which are the open attachment sites
used during implicit repair.

Use explicit attachment tokens when two fragments should be connected by the
serialized tree itself:

```text
[Frag]fragment_A[Attach:0][Frag@0]fragment_B
```

Do not use `[.]` for implicit-anchor prompts:

```text
[Frag]anchor_A[.][Frag]anchor_B
```

That syntax means legacy disconnected components rather than adjacent anchors in
the same repair group.

Decoding repairs implicit-anchor prompts by connecting adjacent root-anchor
components through unused `[Dummy...]` attachment points when possible. If no
valid attachment-based connection is possible, strict decoding raises and
non-strict decoding falls back to the largest valid component.

For linker-design generations that consume every generated-side `[Dummy]` but
still leave the first anchor disconnected, decoding can optionally repair one
missing return attachment. Enable `repair_missing_return_attachment=True` in the
API or `--repair-missing-return-attachment` in the CLI. The repair chooses the
farthest topologically suitable atom from the second anchor, tests the synthetic
single-bond return connection with RDKit sanitization, and leaves decoding
unchanged when no strictly valid site exists.

## Encoding Modes

Default encoding is deterministic for the input order but not canonicalized:

```python
from fragment_selfies import FragmentSelfiesCodec

codec = FragmentSelfiesCodec()
encoded = codec.encode("COC(=O)c1c(N)oc(C)c1C")
```

Canonical encoding gives a stable canonical string:

```python
canonical = codec.encode("COC(=O)c1c(N)oc(C)c1C", canonical=True)
```

Randomized encoding changes atom order and fragment traversal for augmentation:

```python
augmented = codec.encode("COC(=O)c1c(N)oc(C)c1C", randomized=True, seed=0)
```

Randomized encoding also supports implicit-anchor augmentation. With probability
`implicit_probability`, up to `max_implicit_cuts` BRICS edges in each connected component are
serialized as adjacent root anchors instead of an explicit `[Attach:N][Frag@M]`
edge. The default `max_implicit_cuts=1` preserves the two-root implicit behavior;
set it to `2` or higher to produce multi-root implicit samples for multi-fragment
design training:

```python
implicit_augmented = codec.encode(
    "COC(=O)c1c(N)oc(C)c1C",
    randomized=True,
    seed=0,
    implicit_probability=1.0,
    max_implicit_cuts=2,
)
```

The default probability is `0.15` for randomized encoding. Set
`implicit_probability=0.0` to disable implicit-anchor augmentation.

`canonical=True` and `randomized=True` are mutually exclusive.

## CLI Usage

Encode one SMILES string:

```bash
conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --canonical
```

Encode a randomized augmentation:

```bash
conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --randomized \
  --seed 0 \
  --implicit-probability 0.15 \
  --max-implicit-cuts 2
```

Force explicit one-root serialization:

```bash
conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --randomized \
  --style explicit
```

Force implicit adjacent-root serialization when a BRICS edge exists:

```bash
conda run -n fragment_selfies fragment-selfies encode \
  --smiles 'COC(=O)c1c(N)oc(C)c1C' \
  --canonical \
  --style implicit \
  --max-implicit-cuts 2
```

Re-serialize an existing Fragment-SELFIES string into another style:

```bash
conda run -n fragment_selfies fragment-selfies reserialize \
  --fragment-selfies '[Frag][O][=CH0][Branch1][C][Dummy][Dummy][Frag][C][OH0][Dummy]' \
  --style explicit \
  --canonical
```

Decode a compact Fragment-SELFIES string:

```bash
conda run -n fragment_selfies fragment-selfies decode \
  --fragment-selfies '[Frag][C][=C][C][=C][C][=C][Ring1][=Branch1]' \
  --canonical
```

Decoding uses non-strict recovery mode by default for generated strings. Add
`--strict` to require exact Fragment-SELFIES decoding for validation.

For linker-design outputs missing a final return attachment, add the opt-in
repair flag:

```bash
conda run -n fragment_selfies fragment-selfies decode \
  --fragment-selfies '<generated-linker-fragment-selfies>' \
  --repair-missing-return-attachment \
  --canonical
```

## API Usage

```python
from rdkit import Chem
from fragment_selfies import FragmentSelfiesCodec

codec = FragmentSelfiesCodec()

encoded = codec.encode("COC(=O)c1c(N)oc(C)c1C", canonical=True)
mol = codec.decode(encoded)
smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
```

`codec.decode(...)` is non-strict by default. Pass `strict=True` when validation
should fail on malformed Fragment-SELFIES instead of recovering a molecule.
Pass `repair_missing_return_attachment=True` to enable the conservative linker
return-attachment repair described above.

SMILES fragments with dummy attachment atoms can be encoded as a single
Fragment-SELFIES fragment and decoded with dummy atoms preserved:

```python
fragment = codec.encode_fragment("[*]C(=O)O", canonical=True)
fragment_mol = codec.decode_fragment(fragment)
fragment_smiles = Chem.MolToSmiles(fragment_mol, canonical=True, isomericSmiles=False)
```

Use the CLI commands `fragment-selfies encode-fragment` and
`fragment-selfies decode-fragment` for the same fragment-specific conversion.

## Example

Input SMILES:

```text
COC(=O)c1c(N)oc(C)c1C
```

Canonical compact Fragment-SELFIES:

```text
[Frag][C][C][O][C][Branch1][C][N][=C][Branch1][C][Dummy][C][=Ring1][#Branch1][C][Attach:0][Frag@1][O][=C][Branch1][C][Dummy][Dummy][Attach:0][Frag@0][C][O][Dummy][pop][pop]
```

The string has 31 tokens and 3 BRICS fragments. It decodes back to:

```text
COC(=O)c1c(N)oc(C)c1C
```

## Optional Fragment Statistics

Normal compact encode/decode does not need a BRICS fragment vocabulary. The
historical `build-vocab` command remains available as a BRICS fragment counting
and analysis utility.

```bash
conda run --live-stream -n fragment_selfies fragment-selfies build-vocab \
  --input /path/to/molecules.smi \
  --output /path/to/fragment_vocab.jsonl \
  --min-count 1 \
  --workers 8 \
  --canonical \
  --progress
```

## Verification

```bash
conda run --live-stream -n fragment_selfies python -m py_compile src/fragment_selfies/*.py tests/test_fragment_selfies.py
conda run --live-stream -n fragment_selfies python -m pytest -q
```

Current expected result:

```text
37 passed, 1 skipped
```

An optional corpus roundtrip test runs when `FRAGMENT_SELFIES_SAMPLE_SMI` points
to a local `.smi` file.

## Molecular Language Model Integration

A downstream model should train its tokenizer directly on compact Fragment-SELFIES
corpus files. The tokenizer should learn bracketed tokens from the generated
corpus: Fragment-SELFIES structural tokens, SELFIES-compatible fragment-body
tokens, and any task-specific control tokens. No large BRICS fragment-ID
vocabulary is required.

Recommended integration path:

1. Convert molecule corpora from SMILES to compact Fragment-SELFIES.
2. Train a WordLevel tokenizer on the compact Fragment-SELFIES corpus.
3. Use `FragmentSelfiesCodec.encode()` for SMILES to Fragment-SELFIES conversion.
4. Use `FragmentSelfiesCodec.decode()` plus `Chem.MolToSmiles()` for generated molecule recovery.

Fragment-SELFIES is the molecular language used by [Molexar](https://github.com/fairydance/Molexar), a unified multimodal molecular foundation model for drug design.

## Citation

```bibtex
@misc{lin2026molexarunifiedmultimodalmolecular,
      title={Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design}, 
      author={Haoyu Lin and Yiyan Liao and Jinmei Pan and Xinliao Ling and Luhua Lai and Jianfeng Pei},
      year={2026},
      eprint={2606.25865},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2606.25865}, 
}
```

## License

Fragment-SELFIES is released under the MIT License. See `LICENSE` for details.
