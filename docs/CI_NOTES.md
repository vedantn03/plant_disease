# CI Notes

## Artifact-aware behavior

GitHub Actions runs on a clean Linux checkout without the Windows dataset,
`DATA_ROOT`, `mlflow.db`, or `models/best_model.pth`. API integration tests that
need those local artifacts skip with an explicit reason:

- `tests/test_api.py` model-backed endpoint tests
- `tests/test_ood_and_hierarchy.py::test_api_predict_rejects_grey_tile`
- `tests/test_ood_and_hierarchy.py::test_api_predict_crop_constraint`
- `tests/test_ood_and_hierarchy.py::test_predict_batch_endpoint`

Pure heuristic and structural tests continue to run. On the Windows checkout,
where `DATA_ROOT` and the champion checkpoint exist, the same tests run normally.

CI does not validate the API until a fixture-model test exists. That is a plan
for a future change, not a synthetic replacement for the real model.

The workflow installs CPU PyTorch from the official index:
https://download.pytorch.org/whl/cpu
