# CI and release checks

## CI jobs

Pull requests to `main` and pushes to `main` or `master` run separate jobs for lint, tests, secret scanning, Compose configuration, API and UI image builds, and tracked-file checks. Each job reports its own pass or failure. The test job prints `pip freeze` and runs `pytest tests/ -v -rs`, so pytest includes a skip summary.

The fixture-model API test creates randomly initialized weights in a temporary directory and enables that path only when `ENV=test`. It exercises `/health`, `/predict`, and `/metrics` plumbing. It does not test model accuracy and does not use the production checkpoint or dataset.

The tracked-file job rejects tracked `.env`, `.pth`, and `.db` files, plus files larger than the size configured by `check-added-large-files` in `.pre-commit-config.yaml`.

The `detect-private-key` hook blocks common PEM private-key formats. `detect-secrets` scans changed files locally and all tracked files in CI for additional credential patterns with heuristics. These checks can miss some secret formats and may require review of false positives; they are not a guarantee. See the [detect-secrets v1.5.0 documentation](https://github.com/Yelp/detect-secrets/tree/v1.5.0) and its [hook definition](https://raw.githubusercontent.com/Yelp/detect-secrets/v1.5.0/.pre-commit-hooks.yaml).

## Manual GHCR image publishing

The optional publishing workflow is `workflow_dispatch` only, has an explicit `publish` opt-in, and runs only when selected ref is a `v*` tag. It builds and publishes the API image to GHCR; it never copies model weights or dataset files into the image. It needs `packages: write` on `GITHUB_TOKEN` and repository/package access configured to permit the workflow. See [GitHub Actions token permissions](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token) and [GitHub Packages permissions](https://docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages).

GitHub's current billing page says container registry storage and bandwidth are free, with policy changes announced in advance; account-level package quotas for other package types are separate. Review the [GitHub Packages billing terms](https://docs.github.com/en/billing/concepts/product-billing/github-packages) before choosing to publish a private image.

To try the workflow manually, first create and push a disposable test tag with `git tag -a vtest-wp7 -m "Test manual GHCR workflow"` and `git push origin vtest-wp7`, then open **Actions → Publish API image → Run workflow**, choose the tag, and set `publish` to `true`. Do not use a production release tag for this check. Delete the test tag after review if it is no longer needed.

## Inspecting CI results

Open the pull request on GitHub and select **Checks** (or open the repository's **Actions** tab and select the run). Each job has an independent status. A failing job has a red failure mark; expand its failed step to view the command and error output. A skipped test is shown in pytest's `-rs` summary and is not counted as a passed accuracy test.
