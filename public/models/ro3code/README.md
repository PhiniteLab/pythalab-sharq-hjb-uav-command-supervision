# Ro3code Visual Model Assets

This directory contains visual-only aircraft mesh assets used by the React/Three.js frontend demo.

## Provenance

The SAAB Gripen STL files under `public/models/ro3code/saab-gripen/` are sourced from the Ro3code `aircraft_3d_animation` project:

- GitHub repository: https://github.com/Ro3code/aircraft_3d_animation
- Upstream Gripen STL path: https://github.com/Ro3code/aircraft_3d_animation/tree/main/import_stl_model/SAAB-Gripen
- Official release page referenced by the upstream project: https://www.mathworks.com/matlabcentral/fileexchange/86453-aircraft-3d-animation
- Upstream repository license notice: GPL-3.0 (see the upstream `LICENSE`)

For this repository snapshot, the included 13 STL files match the upstream `import_stl_model/SAAB-Gripen/` files exactly at the SHA-256 level.

## Important boundaries

- These meshes are rendering assets only; they are not aerodynamic evidence.
- Backend physics and benchmark results come from `backend/src/uavsim`, not from the visual mesh.
- If a mesh fails to load, the frontend still has a simple placeholder aircraft fallback.
- Upstream provenance and license review for these STL files should be handled separately from the simulation-code claims.
