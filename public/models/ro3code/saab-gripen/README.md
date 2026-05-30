# SAAB Gripen STL Visual Shell

This directory contains the STL mesh parts loaded by `src/components/scene/SimpleFlightScene.tsx` for the visual-only Gripen shell.

## Provenance

These STL files are sourced from the Ro3code `aircraft_3d_animation` repository:

- Repository: https://github.com/Ro3code/aircraft_3d_animation
- Upstream folder: https://github.com/Ro3code/aircraft_3d_animation/tree/main/import_stl_model/SAAB-Gripen
- Related upstream example: https://github.com/Ro3code/aircraft_3d_animation/blob/main/examples/run_animation_gripen.m
- Official release page referenced by the upstream repository: https://www.mathworks.com/matlabcentral/fileexchange/86453-aircraft-3d-animation
- Upstream repository license notice: GPL-3.0 (see the upstream `LICENSE`)

For this repository snapshot, the local STL files match the upstream `import_stl_model/SAAB-Gripen/` STL files exactly at the SHA-256 level.

Expected files:

- `Body.stl`
- `Canopy.stl`, `Canopy_Front.stl`, `Canopy_Rear.stl`
- `FP_Left.stl`, `FP_Right.stl`
- `LE_Left.stl`, `LE_Right.stl`
- `Elevon_Left.stl`, `Elevon_Right.stl`
- `Rudder.stl`
- `AB_Left.stl`, `AB_Right.stl`

These files affect only visualization. They do not change backend simulation, controller benchmarks, experiment result files, or tests.
