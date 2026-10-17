"""Optional compute backends. Each backend module is imported lazily."""


def require_jax():
    """Import and return ``jax`` (with float64 enabled) or raise a helpful ImportError."""
    try:
        import jax
    except ImportError as e:
        raise ImportError(
            "backend='jax' requires the optional dependency JAX. Install it with "
            "`pip install kerneltests[jax]` (or `pip install \"jax[cpu]\"`; see "
            "https://jax.readthedocs.io for GPU wheels), or use backend='numpy'."
        ) from e
    jax.config.update("jax_enable_x64", True)
    return jax
