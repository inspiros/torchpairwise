# Token Sequence Distances

`levenshtein_distances` compares integer token sequences using unit insertion,
deletion and substitution costs. `indel_distances` permits insertion/deletion
only, equivalently substitution costs two. Its value is
`len(x) + len(y) - 2 * LCS_length(x,y)`.

Inputs have shape `[...,N,T]` and `[...,M,U]`; the result is an int64 tensor
`[...,N,M]`. Leading batch axes broadcast. Optional int64 `x1_lengths` and
`x2_lengths` broadcast to each input's sample shape `[...,N]` / `[...,M]`.
Only valid prefixes participate; zero lengths and zero-width sequence axes
are supported. When the second collection is omitted, `x1_lengths` also
supplies its lengths unless `x2_lengths` is explicitly supplied.

Each family has `paired_` and `condensed_` entry points. Paired accepts both
collections with equal sample counts. Condensed accepts one collection and
`x_lengths`, returning its strict upper triangle in row-major `pdist` order.
No dense matrix is constructed for these two modes.

Token IDs are opaque labels: equality matters, their numeric magnitude does
not. Inputs must have matching integer dtype (uint8, int8, int16, int32 or
int64), layout and CPU/CUDA device. They have no derivatives. Tokenization,
vocabularies and any length normalization remain user-owned. These functions
are not corpus-level WER/CER/BLEU/ROUGE evaluators.

The shared C++ implementation uses ATen integer operations, with anti-diagonal dynamic
programming and a workspace proportional to `chunk_size * min(T,U)`.
`chunk_size` bounds pairs across all batches (default 256). Work depends on
padded sequence extents; use reasonably tight padding. The typed
`token_comparisons` operator and C++ entry point execute every comparison mode
without Python recurrence code; see the [native API guide](native-api.md).
No speed advantage is promised over optimized CPU string libraries such as
[RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html).
For short strings or small workloads, CPU comparison can be preferable.
