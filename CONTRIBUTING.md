# Contributing to kerneltests

## Development setup

```bash
git clone <repository-url> && cd kerneltests
python -m venv .venv && source .venv/bin/activate
pip install -e .[test,docs]
```

## Running tests

```bash
pytest --cov=kt --cov-config=.coveragerc     # fails if coverage drops below 90%
mkdocs build --strict                         # docs must build without warnings
```

## The exact-parity requirement for new ports

Every port of a published test must agree **exactly** with the authors' reference implementation for the same seed and inputs, apart from documented bug fixes.

1. Reproduce the reference's random-number protocol: the same generator (`np.random.RandomState(seed)`), the same draws in the same order and with the same shapes.
2. Keep the floating-point evaluation order of the statistic, quantiles and weights. A refactor that is algebraically equal can still change the bits.
3. Add a parity script that runs the reference and the port across every kernel/option combination, both calibration schemes, and null and alternative data. It should compare the decision, statistics, quantiles, p-values and thresholds.
4. Each deviation must be a confirmed reference bug. Document it in `docs/verification.md` and `CHANGELOG.md`, and add a regression test.
5. Add unit tests for power, level under the null, determinism and input validation.

## Style

- Follow PEP 8. Use NumPy-style vectorisation where it doesn't change results.
- Validate inputs with `ValueError` (never `assert`).
- Expose new methods through the high-level entry points in `kt/api.py`, returning `TestResult`.
- Credit the original authors and licences in the module docstring.
- Update `CHANGELOG.md` under *Unreleased*.
