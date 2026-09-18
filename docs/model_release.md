# Sophia Model Release

## License

The Sophia repository, model code, configuration, and released model weights
are intended to be distributed under the Apache License 2.0. See the root
[`LICENSE`](../LICENSE) file. Copyright is held by Arain unless a file or
artifact carries a more specific notice.

This license statement applies to the project and model artifacts; it does not
override the rights of upstream data providers.

## Training data

The pretraining and SFT manifests retain source-level lineage and license
information. Some sources are Creative Commons or permissive software
licenses, while the admitted Chinese web sources include a non-commercial
research-use restriction. The repository license does not grant commercial
rights to those source texts, nor does it authorize redistribution of raw
corpora. Users must inspect the relevant `dataset/**/lineage` records before
redistributing data or using the derived model commercially.

## Public reasoning output

Sophia is released as a thinking model. Public generation preserves the
literal `<think>...</think>` block followed by the answer. The runtime removes
only the terminal EOS marker. Think text is model-generated reasoning, not a
guarantee of correctness, factuality, or an explanation of the actual internal
causal process; downstream applications should treat it as ordinary generated
text and apply their own safety and privacy review.

Provider credentials used for data construction or
judging are runtime secrets and are never part of the release artifacts.

The training workflow stops at a local, lineage-bound release directory; it
does not upload weights or raw data. A maintainer may publish the resulting
artifacts separately after reviewing the evidence and each upstream source
license.

The training identity prompt predates the final source-lineage review and can
occasionally say that all training data is public. That generated statement is
not a license grant. The lineage records and this release notice are the
authoritative description of data rights; if an evaluation exposes the stale
claim, it is reported as a known model limitation rather than silently treated
as legal permission.
