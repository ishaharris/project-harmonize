# Environment and provenance

This repo uses a separate `.venv`, CPython 3.12.12, and `uv.lock`. It does not install model
weights or require local GPU inference. `esm==3.4.1.post1` is an optional dependency; its SDK
is imported only by the Biohub adapter. All comparison code works without it.

The project-local kernel is registered with:

```bash
uv run python -m ipykernel install --sys-prefix --name harmonize --display-name 'Python (Harmonize)'
```

Launch Jupyter through this environment so that it discovers the kernel. Re-register it after
moving the repo or recreating the environment, since kernelspecs contain interpreter paths.

For restricted environments, these optional settings keep caches inside the project:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export MPLCONFIGDIR="$PWD/.tools/matplotlib"
export XDG_CACHE_HOME="$PWD/.tools/cache"
export IPYTHONDIR="$PWD/.tools/ipython"
```

US-align is built by `scripts/bootstrap_native.py` from commit
`fa4376bd99fa17a123d05d7ea47cf6574c80d64f` (20260819). The source archive SHA-256 is
`ee5a2ac8904a3d526a3888d7897a16594de0507bf4b167ec8c0c72b2202776d8`.
The compiler uses `-O3`; no fast-math or platform-specific instruction flags are added.
The executable's checksum and source provenance are stored in `.tools/usalign/provenance.json`;
analysis records the actual binary checksum. US-align's upstream `-v` returns nonzero even
when printing a valid version banner, so version checks inspect the banner explicitly.

The [US-align source](https://github.com/pylelab/USalign) and its upstream license are retained
in the downloaded source tree. The offline browser bundle is 3Dmol 2.5.5 with a shipped license
and SHA-256 provenance in `assets/`. The Python client `py3Dmol` is locked separately.

Provider references:

- [NVIDIA OpenFold2 API](https://docs.api.nvidia.com/nim/reference/openfold-openfold2-infer)
- [NVIDIA MSA search API](https://docs.api.nvidia.com/nim/reference/colabfold-msa-search-infer)
- [RCSB template search](https://search.rcsb.org/)
- [ESM Atlas folding API](https://esmatlas.com/about)
- [Biohub ESM SDK](https://github.com/Biohub/esm)
- [Biotite symmetric lDDT](https://www.biotite-python.org/latest/apidoc/biotite.structure.lddt.html)

The hosted OpenFold2 endpoint does not expose an immutable deployment version in the documented
request. Its identity is recorded with that limitation, alongside endpoint, parameters, SDK
versions, timestamps, checksums, and any request IDs. Stored artifacts reproduce the analysis;
remote inference is not claimed to be bitwise reproducible.
