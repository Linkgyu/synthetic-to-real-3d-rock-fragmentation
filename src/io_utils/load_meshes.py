"""Mesh loading utilities for synthetic rock fragment benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import trimesh


@dataclass(frozen=True)
class FragmentMesh:
    """Container for one rock fragment mesh and its ground-truth metadata."""

    fragment_id: int
    mesh: trimesh.Trimesh
    mesh_path: Path
    volume_m3: float

    @property
    def equivalent_diameter_m(self) -> float:
        """Return equivalent spherical diameter derived from volume."""

        return float((6.0 * self.volume_m3 / 3.141592653589793) ** (1.0 / 3.0))


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh as a `trimesh.Trimesh`, flattening scenes when needed."""

    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)!r}")
    return mesh


def discover_mesh_paths(mesh_dir: Path, suffixes: tuple[str, ...] = (".ply", ".obj", ".stl")) -> list[Path]:
    """Return sorted mesh paths from a directory."""

    mesh_dir = Path(mesh_dir)
    paths: list[Path] = []
    for suffix in suffixes:
        paths.extend(mesh_dir.glob(f"*{suffix}"))
    return sorted(paths)


def load_fragment_meshes(mesh_paths: Iterable[Path]) -> list[FragmentMesh]:
    """Load fragment meshes and assign sequential fragment IDs.

    Volumes are read from the mesh geometry. If a mesh has non-positive or
    invalid volume, its convex hull volume is used as a robust fallback.
    """

    fragments: list[FragmentMesh] = []
    for fragment_id, mesh_path in enumerate(mesh_paths):
        path = Path(mesh_path)
        mesh = _load_single_mesh(path)
        volume = abs(float(mesh.volume))
        if volume <= 0:
            volume = abs(float(mesh.convex_hull.volume))
        fragments.append(
            FragmentMesh(
                fragment_id=fragment_id,
                mesh=mesh,
                mesh_path=path,
                volume_m3=volume,
            )
        )
    return fragments


def fragment_metadata_table(fragments: Iterable[FragmentMesh]) -> pd.DataFrame:
    """Create a fragment metadata table suitable for ground-truth PSD analysis."""

    rows = []
    for fragment in fragments:
        rows.append(
            {
                "fragment_id": fragment.fragment_id,
                "mesh_path": str(fragment.mesh_path),
                "volume_m3": fragment.volume_m3,
                "equivalent_diameter_m": fragment.equivalent_diameter_m,
                "n_vertices": int(len(fragment.mesh.vertices)),
                "n_faces": int(len(fragment.mesh.faces)),
            }
        )
    return pd.DataFrame(rows)

