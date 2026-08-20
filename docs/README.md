# Technical documentation

- `architecture.md`: current task/perception/motion-route architecture.
- `dataset_contracts.md`: training/test datasets and q_opt vs force-adjusted
  supervision.
- `execution_contracts.md`: candidate, IK, collision, execution, and recovery
  requirements; historical route names inside are provenance unless explicitly
  marked as the current CLI.
- `coordinate_and_label_contract.md`: camera, hand-root, and joint-order
  conventions.
- `PROJECT_STRUCTURE.md`: stage ownership reference.
- `history/`: repository reorganization and historical integration notes.
- `WUJI2_OFFICIAL_USD_GRASP_FAILURE_AUDIT.md`: prior official-USD migration
  audit; it is not permission to edit official assets.

For the current runnable system, read the root `README.md`,
`PROJECT_STATUS.md`, and `08_dual_arm_scene_layout/isaaclab_control/MAINLINE.md`
first. Current code and generated reports take precedence over historical
experimental prose.
