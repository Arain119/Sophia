# SM120 Runtime Artifacts

The formal pretraining path on an RTX 5090 uses two CUDA extensions that are
not published as Python dependencies. `install_sm120_runtime` verifies the
installed shared objects by SHA-256 before constructing the graph, so a target
machine must use the qualified builds below.

## Pinned target build

| component | source commit | target build | artifact SHA-256 |
| --- | --- | --- | --- |
| causal-conv1d | `4f6ae4e` | `causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl` | wheel `90142d7710a02947c46d317c91cf608a20a99a465ded1253735f36e23dbe402d`; loaded `causal_conv1d_cuda` `8309d50227c3ee8170225d1616ade91e5c031574e836d0c79f87ac621f0536b9` |
| FlashKDA training extension | `96c4273c814e361a96d49777f8dfae666d21e601` | `flash_kda-0.0.1+96c4273-cp312-cp312-linux_x86_64.whl` | wheel `4c239dd55cff2ddf1b841912b6bdfeb9a07fe89fb9b5654c815c2948b88128cc`; loaded `flash_kda_train_C` `74d5cf52f3d2017f0449d5f6035dcdeec4dbae849f08ac4a1dc210062d397151` |

The matching target environment is Python 3.12, PyTorch `2.8.0+cu128`, CUDA
12.8, and Triton 3.4.0. The FlashKDA wheel is compiled for `sm_120a`; the
causal-conv wheel is the SM120 build, not the older SM89 wheel.

## Rebuild procedure

Keep the source checkouts and wheels on target-native persistent storage (for
example `/root/autodl-tmp/sophia/ops`). Record the full `git rev-parse HEAD`
and the resulting wheel and shared-object hashes in the target-prep output.

```bash
python -m pip install --no-build-isolation -v \
  /root/autodl-tmp/sophia/ops/causal-conv1d-sm120-src

cd /root/autodl-tmp/sophia/ops/flashkda-train-src
FLASH_KDA_CUDA_ARCHS=120a \
  python -m pip wheel --no-build-isolation -v . \
  -w /root/autodl-tmp/sophia/ops/flashkda-train-wheel
python -m pip install --no-deps \
  /root/autodl-tmp/sophia/ops/flashkda-train-wheel/flash_kda-*.whl
```

The build must run with the pinned target stack and CUDA toolkit. Do not
replace a qualified binary with a locally compiled extension without rerunning
the machine recipe and all Muon stability/recapture/resume audits: the binary
hashes are part of the runtime evidence.

The current target already contains both source checkouts, wheels, build logs,
and `direct_url.json` records under `/root/autodl-tmp/sophia/ops`. They are
deployment artifacts rather than model semantics; the signed machine recipe
binds the loaded shared-object hashes.
