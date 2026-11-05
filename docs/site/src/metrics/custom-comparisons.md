<div class="eyebrow">Metric Reference</div>

# Custom Comparisons

Use a built-in comparison when its equation fits. For a user-defined formula,
`make_pairwise` turns a scalar two-sample function into all-pairs, paired,
condensed and single comparisons. Pass the same object to `tp.lazy(x, y, metric)`
to bind the formula to the scalar streaming engine.

For learned representations, separate encoding from comparison. The caller
supplies the encoder and its loaded weights; `make_feature_pairwise` composes
one encoder, or separate query and reference towers, with an explicit head.
Extracted features can feed built-in metrics, custom metrics and streamed operators.

| Extension | What the User Provides | What TorchPairwise Reuses |
|---|---|---|
| [User-defined metric](custom-metrics.md) | A scalar or broadcast-tile forward, optionally a custom VJP, autograd function or `nn.Module` | Pair indexing, modes, tiling, scalar algebra and consumers |
| [Native metric](custom-metrics.md#registered-c-and-cuda-operators) | Registered paired-tile forward/VJP operators | The same source and consumer engine in Python and LibTorch |
| [Encoded features](encoded-features.md) | One encoder or a query/reference pair, and a comparison head | Encoding once before comparison broadcasting; explicit preparation for reuse |

These extensions do not add entries to the built-in comparison catalog. Python
callbacks and encoders keep their own compilation and deployment requirements.
