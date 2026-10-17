# Launch checklist (owner actions)

These are the steps that turn the code into a JMLR MLOSS submission. They need your accounts, so only you can do them.

## Day 1: make it public (about 30 minutes)

1. **Create the GitHub repository.** On github.com, create a public repository named `kerneltests` with no README (we already have one).
2. **Push.** In the `kerneltests` folder:
   ```bash
   git remote add origin https://github.com/<your-username>/kerneltests.git
   git push -u origin main
   ```
3. **Check CI.** Open the Actions tab. Fix anything red on macOS, Windows or Python 3.9; these have never run.
4. **Turn on GitHub Pages for the docs.** `pip install -e .[docs] && mkdocs gh-deploy`, then replace `<repository-url>` in README.md and the docs with the real URLs and commit.
5. **Enable Issues and Discussions** in the repository settings.

## Day 2: release (about 30 minutes)

6. **Reserve the name on PyPI.** Check that `kerneltests` is free on pypi.org. If it's taken, choose a new name before anyone installs it (it appears in `pyproject.toml`, the README and the docs).
7. **Publish.** Create a PyPI account with 2FA, then:
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```
   Better: set up PyPI "trusted publishing" from GitHub Actions so releases come from tagged commits.
8. **Tag the release** `v0.1.0` on GitHub with release notes taken from CHANGELOG.md.

## Week 1: engage the original author

9. **File the three issues** in `UPSTREAM_ISSUES.md` on `antoninschrab/mmdagg`, `antoninschrab/ksdagg` and `antoninschrab/agginc`. Open a pull request with the fix for the one-line bugs (the `p_vals[idx]` indexing, the missing `import warnings`, the `bandwidths[1]` check). Merged PRs are public evidence of engagement.
10. **Email Antonin Schrab** with the link: the five bugs, the exact-parity verification, and the question of whether he'd like to link the library from his repositories or advise on it. A link from the original READMEs is the single best source of early users.

## Months 1–6: users (the deciding factor for the MLOSS track)

The track asks for "evidence of an active user community". Track these numbers from day one:

- **Downloads:** pypistats.org/packages/kerneltests
- **GitHub:** stars, forks, issues opened by people other than you, external pull requests
- **Dependents:** projects and papers that use or cite the library

Ways to earn real users, roughly in order of effect:

1. Get linked from the original method repositories (step 10).
2. Write two or three tutorials on real problems: detecting distribution shift between training and deployment data, checking a generative model's samples against real data, testing independence of features. Post them where practitioners read (a blog, Towards Data Science, r/MachineLearning, relevant Discords).
3. Answer existing questions on Stack Overflow and Cross Validated about two-sample, independence and goodness-of-fit tests in Python, pointing to the library where it genuinely fits.
4. Present a lightning talk at a local Python or ML meetup, or submit a short talk to a SciPy, PyData or EuroSciPy conference.
5. Respond to every issue quickly. Responsive maintenance is what turns first users into repeat users and contributors.

## Submission (when the numbers justify it)

11. Finish `paper/kerneltests_mloss.tex` (every TODO), move it to JMLR's `jmlr2e` style, and submit through the JMLR submission system as an MLOSS paper. The cover letter must state the license, project URL, the version reviewed, and evidence of the user community.
