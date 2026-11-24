"""Nonnegative spectral-factorisation part extraction prototype.

The package deliberately implements its own STFT, NMF and masking so that every
number reported by the UI/evaluation can be traced to source here.  There is no
call out to an external source-separation model.
"""

# 0.2.0 — linear-algebra backend migration: the NMF matrix solves (SVD, NNLS
# lstsq, GEMM products, reductions, column normalisation) run through a
# selectable backend (numpy reference / scipy LAPACK-BLAS alternative); see
# nmfsep.linalg and nmfsep.equivalence.  Runs record "linalg_backend" in their
# diagnostics/config so the migration version is always attributable.
__version__ = "0.2.0"
