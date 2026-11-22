"""Contract tests for the quad remeshing helpers.

Every test is deterministic (no randomness, no clock, no file system) and uses
only the standard library plus pytest.  Triangle records are built inline with
``{"v": (i, j, k), "p": (...), "uv": (...)}`` as documented in
``quadremesh/pair.py``.
"""

import math

import pytest

from quadremesh.pair import (
    build_adjacency,
    fill_leftovers,
    merge_to_quad,
    pair_score,
    pair_triangles,
    preserve_seams,
    remesh_batch,
    validate_quads,
)

EPS = 1e-9


# ---------------------------------------------------------------------------
# fixtures as plain data (no stub is called while building them)
# ---------------------------------------------------------------------------


def _tri(positions, i, j, k, uv=None):
    corners = (i, j, k)
    if uv is None:
        uv = tuple((positions[v][0], positions[v][1]) for v in corners)
    return {
        "v": corners,
        "p": tuple(positions[v] for v in corners),
        "uv": tuple(uv),
    }


def _grid(nx, ny, scale=1.0, diagonal="ac"):
    """Return ``(positions, tris)`` for an ``nx`` by ``ny`` cell grid in z=0.

    Every cell is split along ``"ac"`` or ``"bd"``; both splits keep the two
    triangles counter-clockwise when seen from +z, so the quad normal must be
    +z too.
    """
    positions = []
    for y in range(ny + 1):
        for x in range(nx + 1):
            positions.append((x * scale, y * scale, 0.0))

    def vid(x, y):
        return y * (nx + 1) + x

    tris = []
    for y in range(ny):
        for x in range(nx):
            a, b = vid(x, y), vid(x + 1, y)
            c, d = vid(x + 1, y + 1), vid(x, y + 1)
            if diagonal == "ac":
                corners = ((a, b, c), (a, c, d))
            else:
                corners = ((a, b, d), (b, c, d))
            for v in corners:
                uv = tuple((positions[i][0], positions[i][1]) for i in v)
                tris.append(
                    {"v": v, "p": tuple(positions[i] for i in v), "uv": uv}
                )
    return positions, tris


# ---------------------------------------------------------------------------
# geometry helpers (pure functions, independent of the module under test)
# ---------------------------------------------------------------------------


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _length(v):
    return math.sqrt(_dot(v, v))


def _unit(v):
    n = _length(v)
    if n <= EPS:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _face_normal(tri):
    p = tri["p"]
    return _unit(_cross(_sub(p[1], p[0]), _sub(p[2], p[0])))


def _newell(points):
    nx = ny = nz = 0.0
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx, ny, nz)


def _newell_normal(points):
    return _unit(_newell(points))


def _quad_area(points):
    return 0.5 * _length(_newell(points))


def _signed_area_xy(points):
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)


def _is_simple_quad(points):
    flat = [(p[0], p[1]) for p in points]
    if _segments_cross(flat[0], flat[1], flat[2], flat[3]):
        return False
    if _segments_cross(flat[1], flat[2], flat[3], flat[0]):
        return False
    return True


def _coverage_counts(quads, tris):
    counts = [0] * len(tris)
    for quad in quads:
        corners = set(quad)
        for t, tri in enumerate(tris):
            if set(tri["v"]) <= corners:
                counts[t] += 1
    return counts


def _coverage(quads, tris):
    counts = _coverage_counts(quads, tris)
    return sum(1 for c in counts if c == 1) / len(tris)


# ---------------------------------------------------------------------------
# defect 1: ordered-pair edge keys break non-manifold adjacency
# ---------------------------------------------------------------------------


def test_build_adjacency_keeps_non_manifold_edge_in_one_entry():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    tris = [
        _tri(positions, 0, 1, 2),
        _tri(positions, 1, 0, 3),
        _tri(positions, 0, 1, 4),
    ]
    adj = build_adjacency(tris)

    assert adj[(0, 1)] == (0, 1, 2)
    assert adj[(1, 2)] == (0,)
    assert adj[(0, 2)] == (0,)
    assert adj[(0, 3)] == (1,)
    assert adj[(0, 4)] == (2,)
    assert (1, 0) not in adj
    assert (2, 1) not in adj
    assert len(adj) == 7


# ---------------------------------------------------------------------------
# defect 2: coplanarity only, no area / edge-length term
# ---------------------------------------------------------------------------


def test_pair_score_penalizes_sliver_quads_not_only_coplanarity():
    square_positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    a = _tri(square_positions, 0, 1, 2)
    b = _tri(square_positions, 1, 3, 2)
    square = pair_score(a, b)
    assert square == pytest.approx(1.0, abs=1e-9)

    sliver_positions = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (5.0, 0.02, 0.0),
        (5.0, -0.02, 0.0),
    ]
    thin_a = _tri(sliver_positions, 0, 1, 2)
    thin_b = _tri(sliver_positions, 1, 0, 3)
    sliver = pair_score(thin_a, thin_b)

    assert sliver == pytest.approx(pair_score(thin_b, thin_a), abs=1e-9)
    assert sliver < square - 0.2
    assert sliver < 0.75

    scaled = [(x * 1000.0, y * 1000.0, z * 1000.0) for x, y, z in sliver_positions]
    scaled_a = _tri(scaled, 0, 1, 2)
    scaled_b = _tri(scaled, 1, 0, 3)
    assert pair_score(scaled_a, scaled_b) == pytest.approx(sliver, abs=1e-9)


# ---------------------------------------------------------------------------
# defect 3: first match instead of the globally highest score
# ---------------------------------------------------------------------------


def test_pair_triangles_takes_highest_score_not_first_match():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.5, -0.02, 0.0),
    ]
    # Triangle 2 has two candidate partners: 0 (low score, first in the list)
    # and 1 (a unit-square diagonal, score 1.0).
    low = _tri(positions, 1, 0, 4)
    high = _tri(positions, 1, 3, 2)
    centre = _tri(positions, 0, 1, 2)
    tris = [low, high, centre]

    adj = build_adjacency(tris)
    assert pair_score(tris[1], tris[2]) > pair_score(tris[0], tris[2])

    pairs = pair_triangles(tris, adj, 0.0)
    assert pairs == [(1, 2)]


# ---------------------------------------------------------------------------
# defect 4: clockwise quad winding
# ---------------------------------------------------------------------------


def test_merge_to_quad_winding_is_counter_clockwise():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    ]
    a = _tri(positions, 0, 1, 2)
    b = _tri(positions, 1, 3, 2)
    quad = merge_to_quad(a, b)

    assert quad == (0, 1, 3, 2)
    points = [positions[i] for i in quad]
    normal_dot = _dot(_face_normal(a), _newell_normal(points))
    assert normal_dot > 0.999
    assert _signed_area_xy(points) > 0.0


# ---------------------------------------------------------------------------
# defect 5: shared-edge corners not aligned, quad self-intersects
# ---------------------------------------------------------------------------


def test_merge_to_quad_aligns_rotated_shared_edge_without_self_intersection():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    ]
    a = _tri(positions, 0, 1, 2)
    # Same geometry as (1, 3, 2), but rotated so the shared edge (1, 2) is the
    # last two corners; a merge that assumes corner order folds the quad.
    b = _tri(positions, 3, 2, 1)

    quad = merge_to_quad(a, b)
    assert quad == (0, 1, 3, 2)
    assert len(set(quad)) == 4
    points = [positions[i] for i in quad]
    assert _is_simple_quad(points)
    assert _dot(_face_normal(a), _newell_normal(points)) > 0.999
    assert validate_quads([quad], positions) == []


# ---------------------------------------------------------------------------
# defect 6: leftover triangles are dropped, coverage < 1
# ---------------------------------------------------------------------------


def test_fill_leftovers_keeps_unpaired_triangles_as_degenerate_quads():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.5, -1.0, 0.0),
    ]
    tris = [
        _tri(positions, 0, 1, 2),
        _tri(positions, 1, 3, 2),
        _tri(positions, 1, 0, 4),
    ]
    adj = build_adjacency(tris)

    pairs = pair_triangles(tris, adj, 0.0)
    assert pairs == [(0, 1)]

    quads = [merge_to_quad(tris[i], tris[j]) for i, j in pairs]
    out = fill_leftovers(quads, tris, adj)

    assert len(out) == 2
    assert out[0] == quads[0]
    assert out[1] == (1, 0, 4, 1)
    assert _coverage_counts(out, tris) == [1, 1, 1]
    assert _coverage(out, tris) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# defect 7: seams compare vertex indices only, not UVs
# ---------------------------------------------------------------------------


def test_preserve_seams_splits_uv_discontinuity_with_welded_indices():
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (10.0, 0.0, 0.0),
        (11.0, 0.0, 0.0),
        (10.0, 1.0, 0.0),
        (11.0, 1.0, 0.0),
    ]
    uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    t0 = _tri(positions, 0, 1, 2, uv=(uv[0], uv[1], uv[2]))
    t1 = _tri(positions, 1, 3, 2, uv=(uv[1], uv[3], uv[2]))
    t2 = _tri(positions, 4, 5, 6, uv=(uv[0], uv[1], uv[2]))
    # Same welded indices 5 and 6, but the UVs on this side are unfolded.
    t3 = _tri(
        positions,
        5,
        7,
        6,
        uv=((uv[1][0] + 0.5, uv[1][1]), (uv[3][0] + 0.5, uv[3][1]), (uv[2][0] + 0.5, uv[2][1])),
    )

    mesh = {"tris": [t0, t1, t2, t3], "positions": positions, "threshold": 0.0}
    result = preserve_seams(mesh, [])

    assert result["pairs"] == [(0, 1)]
    assert result["triangle_count"] == 4
    assert result["coverage"] == pytest.approx(1.0, abs=1e-9)
    assert result["quads"] == [
        (0, 1, 3, 2),
        (4, 5, 6, 4),
        (5, 7, 6, 5),
    ]

    # A seam listed explicitly blocks pairing even when the UVs agree.
    explicit = preserve_seams(
        {"tris": [t0, t1], "positions": positions, "threshold": 0.0}, [(2, 1)]
    )
    assert explicit["pairs"] == []
    assert explicit["quads"] == [(0, 1, 2, 0), (1, 3, 2, 1)]
    assert explicit["coverage"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# defect 8: bow-tie quads are not reported
# ---------------------------------------------------------------------------


def test_validate_quads_reports_self_intersecting_bowtie():
    positions = [
        (0.0, 0.0, 0.0),
        (3.0, 2.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.5, 1.5, 0.0),
    ]
    messages = validate_quads([(0, 1, 2, 3)], positions)

    assert isinstance(messages, list)
    assert messages
    assert all(isinstance(m, str) for m in messages)
    assert any(m.startswith("SELF_INTERSECTION") for m in messages)

    good = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    assert validate_quads([(0, 1, 2, 3)], good) == []


# ---------------------------------------------------------------------------
# defect 9: adjacency table reused across meshes
# ---------------------------------------------------------------------------


def test_remesh_batch_does_not_share_adjacency_between_meshes():
    pos_a, tris_a = _grid(3, 3, diagonal="ac")
    pos_b, tris_b = _grid(2, 2, diagonal="bd")

    mesh_a = {"tris": tris_a, "positions": pos_a, "threshold": 0.0}
    mesh_b = {"tris": tris_b, "positions": pos_b, "threshold": 0.0}

    batch = remesh_batch([mesh_a, mesh_b], {"threshold": 0.0})
    assert len(batch["meshes"]) == 2
    assert [r["triangle_count"] for r in batch["meshes"]] == [
        len(tris_a),
        len(tris_b),
    ]

    solo_a = remesh_batch([mesh_a], {"threshold": 0.0})["meshes"][0]
    solo_b = remesh_batch([mesh_b], {"threshold": 0.0})["meshes"][0]
    assert batch["meshes"][0] == solo_a
    assert batch["meshes"][1] == solo_b
    assert batch["meshes"][0]["pairs"] != batch["meshes"][1]["pairs"]


# ---------------------------------------------------------------------------
# defect 10: absolute threshold, scale dependent pairing
# ---------------------------------------------------------------------------


def test_pair_triangles_threshold_is_normalized_and_scale_invariant():
    positions, tris = _grid(2, 2)
    adj = build_adjacency(tris)
    base = pair_triangles(tris, adj, 0.9)
    assert base

    for factor in (1e-3, 1e3):
        scaled_positions = [
            (x * factor, y * factor, z * factor) for x, y, z in positions
        ]
        scaled_tris = [
            _tri(scaled_positions, *tri["v"]) for tri in tris
        ]
        scaled_adj = build_adjacency(scaled_tris)
        assert pair_triangles(scaled_tris, scaled_adj, 0.9) == base


# ---------------------------------------------------------------------------
# extra: regular grid pairing rate >= 90%
# ---------------------------------------------------------------------------


def test_regular_grid_pairing_rate_is_at_least_90_percent():
    positions, tris = _grid(8, 8)
    adj = build_adjacency(tris)

    pairs = pair_triangles(tris, adj, 0.5)
    assert 2 * len(pairs) / len(tris) >= 0.9

    used = [index for pair in pairs for index in pair]
    assert len(used) == len(set(used))


# ---------------------------------------------------------------------------
# extra: every emitted quad has positive area
# ---------------------------------------------------------------------------


def test_every_output_quad_has_positive_area():
    positions, tris = _grid(4, 3)
    adj = build_adjacency(tris)

    pairs = pair_triangles(tris, adj, 0.5)
    quads = [merge_to_quad(tris[i], tris[j]) for i, j in pairs]
    out = fill_leftovers(quads, tris, adj)

    assert out
    for quad in out:
        assert _quad_area([positions[i] for i in quad]) > EPS
    assert validate_quads(out, positions) == []


# ---------------------------------------------------------------------------
# extra: quad winding agrees with the source triangles
# ---------------------------------------------------------------------------


def test_quad_winding_matches_source_triangle_normals():
    positions, tris = _grid(3, 3)
    adj = build_adjacency(tris)

    pairs = pair_triangles(tris, adj, 0.5)
    assert len(pairs) >= 8

    for i, j in pairs:
        quad = merge_to_quad(tris[i], tris[j])
        normal = _newell_normal([positions[v] for v in quad])
        assert _dot(_face_normal(tris[i]), normal) > 0.99
        assert _dot(_face_normal(tris[j]), normal) > 0.99


# ---------------------------------------------------------------------------
# extra: coverage is exactly 1
# ---------------------------------------------------------------------------


def test_coverage_is_exactly_one_no_triangle_lost():
    positions, tris = _grid(5, 4)
    adj = build_adjacency(tris)

    pairs = pair_triangles(tris, adj, 0.5)
    quads = [merge_to_quad(tris[i], tris[j]) for i, j in pairs]
    out = fill_leftovers(quads, tris, adj)

    counts = _coverage_counts(out, tris)
    assert counts.count(1) == len(tris)
    assert _coverage(out, tris) == 1.0
    assert len(out) >= len(tris) // 2


# ---------------------------------------------------------------------------
# extra: two meshes remesh independently, same as single runs
# ---------------------------------------------------------------------------


def test_two_meshes_remesh_independently_like_single_runs():
    pos_1, tris_1 = _grid(2, 2, scale=1.0)
    pos_2, tris_2 = _grid(4, 2, scale=0.01)
    pos_3, tris_3 = _grid(1, 3, scale=1000.0)

    meshes = [
        {"tris": tris_1, "positions": pos_1, "threshold": 0.0},
        {"tris": tris_2, "positions": pos_2, "threshold": 0.0},
        {"tris": tris_3, "positions": pos_3, "threshold": 0.0},
    ]

    batch = remesh_batch(meshes, {"threshold": 0.0})
    assert len(batch["meshes"]) == 3
    assert [r["triangle_count"] for r in batch["meshes"]] == [8, 16, 6]

    for mesh, result in zip(meshes, batch["meshes"]):
        solo = remesh_batch([mesh], {"threshold": 0.0})["meshes"][0]
        assert result == solo
