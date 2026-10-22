# Known Limitations

- Source queue mappings are proxy labels, not independent ground truth. Synthetic corruption only measures recovery of injected errors.
- No human-review credentials, independently adjudicated test labels, or source-license verification evidence was available in the repository.
- Robust-learning comparisons are validation-only and require a fresh pipeline run to regenerate their artifact after this audit.
- Dashboard automated tests currently validate artifact loading and privacy masking, not browser interactions.
- The full workflow writes large derived CSV/model artifacts in place. Run it only when those outputs may be refreshed.
