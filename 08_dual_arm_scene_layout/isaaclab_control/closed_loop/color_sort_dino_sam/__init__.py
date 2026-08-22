"""DINO+SAM multi-object color-sort front-end.

Frozen semantics:
- SourceZone decides target eligibility.
- GraspContextZone supplies DGN2 environment context.
- GroundingDINO + SAM is the only color/object perception path.
- HSV is not used.
- Object labels are capture-local only.
"""
