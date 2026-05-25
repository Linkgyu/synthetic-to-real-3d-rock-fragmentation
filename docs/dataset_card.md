# Dataset Card

## Dataset Name

Synthetic-to-real 3D rock fragmentation benchmark.

## Dataset Type

Synthetic mesh and point-cloud benchmark for rock fragment segmentation and PSD estimation.

## Intended Use

- Prototype LiDAR point-cloud fragmentation analysis.
- Evaluate segmentation and size-estimation bias under controlled degradation.
- Analyse P80 error sensitivity to point density, noise, viewpoint count, and occlusion.

## Ground Truth

Synthetic fragment meshes provide ground-truth fragment IDs and volumes. Equivalent spherical diameter is computed from each fragment volume.

## Known Limitations

- Synthetic fragment geometry may not match all field muckpile morphology.
- Virtual LiDAR is simplified and does not model all sensor-specific effects.
- Segmentation baselines are classical and not expected to solve heavily occluded scenes.
- Results should not be presented as field-validated without real scan comparison.

## Recommended Metadata

Each benchmark sample should store:

- pile ID
- fragment ID
- mesh path
- fragment volume
- equivalent spherical diameter
- scan viewpoint
- point density
- noise level
- occlusion model settings
- segmentation parameters

