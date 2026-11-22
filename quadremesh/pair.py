"""Quad-dominant remeshing helpers for triangle meshes.

Shared contract for every public function below:

* A triangle is a mapping with exactly these keys::

      {
          "v":  (i, j, k),                            # vertex indices
          "p":  ((x, y, z), (x, y, z), (x, y, z)),    # positions of i, j, k
          "uv": ((u, v), (u, v), (u, v)),             # UVs of i, j, k
      }

  ``"p"`` and ``"uv"`` are in the same corner order as ``"v"``: corner ``t``
  owns vertex ``"v"[t]`` and carries the attributes ``"p"[t]`` and ``"uv"[t]``.
* A quad is a 4-tuple of vertex indices in counter-clockwise order.  A
  triangle that is not merged is stored as the degenerate quad ``(a, b, c, a)``
  (the first index is repeated once at the end).
* Half-edge indexing: the undirected edge between vertices ``i`` and ``j`` is
  keyed by ``(min(i, j), max(i, j))``, never by an ordered pair.  Every
  incidence query is answered from these normalized keys.
* Coordinates use a right-handed system; a triangle is counter-clockwise seen
  from its front side, so its unit normal is
  ``normalize((p1 - p0) x (p2 - p0))``.  Winding is what makes the front side
  visible: reversing a quad puts it behind the back-face cull.
* Floating-point comparisons use an absolute tolerance of ``1e-9``.

Everything here is pure, deterministic and offline: no randomness, no file or
network access, standard library only.
"""

EPS = 1e-9


# ---------------------------------------------------------------------------
# small vector helpers
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
    return _dot(v, v) ** 0.5


def _unit(v):
    n = _length(v)
    if n <= EPS:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _face_normal(p):
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


def _edge_key(i, j):
    return (i, j) if i <= j else (j, i)


def _shared_edge(a, b):
    """Return ``(apex_a, u, apex_b, v)`` for the shared edge of ``a``/``b``.

    ``(u, v)`` is the shared edge in the direction ``a`` traverses it.  Returns
    ``None`` when the triangles do not share exactly one undirected edge.
    """
    va, vb = a["v"], b["v"]
    shared = set(va) & set(vb)
    if len(shared) != 2:
        return None
    u = v = None
    for c in range(3):
        i, j = va[c], va[(c + 1) % 3]
        if i in shared and j in shared:
            u, v = i, j
            break
    if u is None:
        return None
    apex_a = next(x for x in va if x not in shared)
    apex_b = next(x for x in vb if x not in shared)
    return apex_a, u, apex_b, v


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------


def build_adjacency(tris) -> dict:
    """Return the undirected edge -> incident triangle indices map.

    For every triangle index ``t`` and every corner ``c`` in ``0, 1, 2`` the
    edge between ``tris[t]["v"][c]`` and ``tris[t]["v"][(c + 1) % 3]`` is
    recorded under the normalized key ``(min(i, j), max(i, j))``.  The value is
    the *tuple* of all triangle indices that contain that edge, sorted
    ascending.

    Consequences that the tests rely on: a boundary edge maps to a 1-tuple; an
    interior edge maps to a 2-tuple; a non-manifold edge shared by three (or
    more) triangles maps to one single 3-or-more tuple.  The edge is never
    split into two keys such as ``(0, 1)`` and ``(1, 0)``, and a non-manifold
    edge is not truncated to the first two incident triangles.  A triangle
    with a repeated vertex still contributes its (collapsed) edges to the same
    keys.

    The returned dict contains every edge of every triangle exactly once, and
    triangle indices are positions in ``tris``.

    Conventions: half-edge keys are ``(min, max)``, tolerance 1e-9.
    """
    incidence = {}
    for t, tri in enumerate(tris):
        v = tri["v"]
        for c in range(3):
            key = _edge_key(v[c], v[(c + 1) % 3])
            incidence.setdefault(key, []).append(t)
    return {key: tuple(sorted(ts)) for key, ts in incidence.items()}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def pair_score(a, b) -> float:
    """Return the merge quality of two triangles as a number in ``[0, 1]``.

    ``a`` and ``b`` are triangle mappings.  When they do not share exactly one
    undirected edge the score is ``0.0``.  Otherwise::

        coplanar = clamp(0.5 * (1.0 + dot(n_a, n_b)), 0.0, 1.0)
        quad     = (apex_a, u, apex_b, v)          # see merge_to_quad
        area     = abs(Newell(quad)) / 2.0
        longest  = max(|q1 - q0|, |q2 - q1|, |q3 - q2|, |q0 - q3|)
        shape    = clamp(area / longest ** 2, 0.0, 1.0)
        score    = 0.5 * coplanar + 0.5 * shape

    ``n_a`` / ``n_b`` are the unit face normals and ``(u, v)`` is the shared
    edge as traversed by ``a``; ``apex_a`` / ``apex_b`` are the two remaining
    corners.  The merged boundary has four outer edges, and the shared edge
    itself is a diagonal, so it is not one of them.

    The ``shape`` term is why two coplanar triangles must not score like a
    square: a long thin merged quad has a small area relative to the square of
    its longest edge and scores low.  Coplanarity alone is not the score.
    The score is symmetric, ``pair_score(a, b) == pair_score(b, a)`` for
    consistently wound inputs, and scale invariant: multiplying every
    coordinate by ``k > 0`` changes it by at most ``1e-9``.

    Conventions: half-edge keys are ``(min, max)``, tolerance 1e-9.
    """
    info = _shared_edge(a, b)
    if info is None:
        return 0.0
    apex_a, u, apex_b, v = info

    pos = {}
    for idx, p in zip(a["v"], a["p"]):
        pos.setdefault(idx, p)
    for idx, p in zip(b["v"], b["p"]):
        pos.setdefault(idx, p)
    quad_pts = [pos[apex_a], pos[u], pos[apex_b], pos[v]]

    n_a = _face_normal(a["p"])
    n_b = _face_normal(b["p"])
    coplanar = _clamp(0.5 * (1.0 + _dot(n_a, n_b)))

    area = 0.5 * _length(_newell(quad_pts))
    longest = max(
        _length(_sub(quad_pts[(i + 1) % 4], quad_pts[i])) for i in range(4)
    )
    shape = _clamp(area / (longest * longest)) if longest > EPS else 0.0

    return 0.5 * coplanar + 0.5 * shape


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def pair_triangles(tris, adj, threshold) -> list[tuple]:
    """Pair triangles greedily, always taking the globally highest score first.

    ``adj`` is :func:`build_adjacency` of ``tris``.  Candidates are the edges
    whose incidence tuple has exactly two triangle indices; boundary edges
    (1 incident triangle) and non-manifold edges (3 or more) never produce a
    pair.  ``threshold`` is normalized to ``[0.0, 1.0]``: ``0.0`` accepts every
    candidate and a candidate is eligible when its :func:`pair_score` is
    greater than or equal to the threshold, compared with tolerance ``1e-9``.

    Repeat until no candidate remains: take the eligible candidate with the
    highest score; ties are broken by the smaller ``(i, j)``.  If both of its
    triangles are still unpaired, accept it.  This is a greedy maximum-score
    matching, not a first-come-first-served scan: a triangle whose best partner
    appears late in the triangle list still gets that best partner, and a low
    score seen early must not block a higher score seen later.

    Return the accepted pairs as ``(i, j)`` with ``i < j``, sorted by ``i``
    then ``j``.  Every triangle appears in at most one pair.  Missing or
    malformed entries in ``adj`` are not repaired.

    Conventions: half-edge keys are ``(min, max)``, threshold is normalized to
    ``[0, 1]``, tolerance 1e-9.
    """
    threshold = _clamp(float(threshold))

    candidates = []
    for inc in adj.values():
        if len(inc) != 2:
            continue
        i, j = inc
        if i == j:
            continue
        if i > j:
            i, j = j, i
        score = pair_score(tris[i], tris[j])
        if score >= threshold - EPS:
            candidates.append((-score, i, j))
    # Highest score first; ties broken by the smaller (i, j).
    candidates.sort()

    used = set()
    pairs = []
    for _neg_score, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def merge_to_quad(a, b) -> tuple:
    """Merge two edge-adjacent triangles into one counter-clockwise quad.

    The triangles must share exactly one undirected edge.  Let ``(u, v)`` be
    that edge in the direction in which ``a`` traverses it: because ``a`` is
    counter-clockwise, ``u -> v`` is one of its three directed corners.  Let
    ``apex_a`` be a's corner that is not on the shared edge and ``apex_b`` the
    same for ``b``.  The returned quad is::

        (apex_a, u, apex_b, v)

    so the shared edge becomes a diagonal.  Its four boundary edges are
    ``apex_a -> u``, ``u -> apex_b``, ``apex_b -> v`` and ``v -> apex_a``; the
    shared edge must not survive on the boundary, because that is what folds a
    bow-tie into the result.

    Corner order is searched rather than assumed: a triangle whose vertex tuple
    starts on another corner (rotated, e.g. ``(c, a, b)``) merges to the same
    quad.  ``b`` has to traverse the shared edge as ``v -> u``; when it
    traverses ``u -> v`` the two inputs disagree on the front side and
    ``ValueError`` is raised.  ``ValueError`` is also raised when the triangles
    share zero edges or more than one edge.

    The quad is counter-clockwise seen from the same side as ``a``: its Newell
    normal has a positive dot product with a's face normal, so back-face
    culling still keeps it visible.

    Conventions: quad winding is counter-clockwise, tolerance 1e-9.
    """
    info = _shared_edge(a, b)
    if info is None:
        raise ValueError("triangles must share exactly one undirected edge")
    apex_a, u, apex_b, v = info

    vb = b["v"]
    traverses_vu = any(
        vb[c] == v and vb[(c + 1) % 3] == u for c in range(3)
    )
    if not traverses_vu:
        raise ValueError("triangles disagree on the front side (winding)")
    return (apex_a, u, apex_b, v)


# ---------------------------------------------------------------------------
# leftovers
# ---------------------------------------------------------------------------


def fill_leftovers(quads, tris, adj) -> list:
    """Complete a quad list so that every triangle is covered exactly once.

    ``quads`` are 4-tuples already produced by :func:`merge_to_quad`, ``tris``
    is the full triangle list.  A triangle is covered by a quad when all three
    of its vertex indices occur among the quad's four indices; a merged quad
    covers exactly two triangles and a degenerate quad covers one.

    Input quads keep their original order and are never rewritten.  Every
    triangle that no input quad covers is appended, in ascending triangle
    index, as the degenerate quad ``(a, b, c, a)`` -- same winding, first
    vertex repeated at the end.  No triangle may be dropped: the returned list
    covers every triangle exactly once, so coverage is exactly ``1.0``, and
    ``len(returned) == len(quads) + len(leftovers)``.

    ``adj`` is accepted for symmetry with the rest of the pipeline and must not
    change the result.

    Conventions: leftovers are ``(a, b, c, a)``, tolerance 1e-9.
    """
    covered = [False] * len(tris)
    for quad in quads:
        corners = set(quad)
        for t, tri in enumerate(tris):
            if not covered[t] and set(tri["v"]) <= corners:
                covered[t] = True

    out = list(quads)
    for t, tri in enumerate(tris):
        if not covered[t]:
            a, b, c = tri["v"]
            out.append((a, b, c, a))
    return out


# ---------------------------------------------------------------------------
# seams
# ---------------------------------------------------------------------------


def preserve_seams(mesh, seams) -> dict:
    """Remesh one mesh without ever merging across a UV seam.

    ``mesh`` maps ``"tris"`` to the triangle list and ``"positions"`` to the
    vertex positions; it may also carry ``"threshold"`` (default ``0.5``).
    ``seams`` is an iterable of undirected vertex-index pairs, in either
    direction, that must stay split.

    A candidate edge is a seam when it is listed in ``seams`` *or* when the two
    incident triangles disagree on the UV of a shared corner by more than
    ``1e-9`` in ``u`` or in ``v``.  Index equality alone is not enough: a
    welded mesh can reuse the very same vertex indices on both sides of an
    unfolded UV border, and that border still must not be merged across.
    Seam edges are removed from the candidate set before pairing.  Everything
    else uses the greedy maximum-score matching of :func:`pair_triangles` with
    the mesh threshold, then :func:`merge_to_quad`, then
    :func:`fill_leftovers`.

    Return ``{"pairs": [(i, j), ...], "quads": [...], "triangle_count": n,
    "coverage": 1.0}``.  ``pairs`` are sorted ``(i, j)`` with ``i < j``; no
    returned pair shares a seam edge; ``quads`` covers every triangle exactly
    once so ``coverage`` is the fraction covered and equals ``1.0``.

    Conventions: half-edge keys are ``(min, max)``, UV tolerance 1e-9.
    """
    tris = mesh["tris"]
    threshold = mesh.get("threshold", 0.5)

    adj = build_adjacency(tris)

    seam_edges = set()
    for pair in seams:
        i, j = pair
        seam_edges.add(_edge_key(i, j))

    for key, inc in adj.items():
        if len(inc) != 2 or key in seam_edges:
            continue
        t1, t2 = tris[inc[0]], tris[inc[1]]
        uv1 = dict(zip(t1["v"], t1["uv"]))
        uv2 = dict(zip(t2["v"], t2["uv"]))
        for vtx in key:
            a = uv1.get(vtx)
            b = uv2.get(vtx)
            if a is None or b is None:
                continue
            if abs(a[0] - b[0]) > EPS or abs(a[1] - b[1]) > EPS:
                seam_edges.add(key)
                break

    candidates = {k: v for k, v in adj.items() if k not in seam_edges}
    pairs = pair_triangles(tris, candidates, threshold)

    quads = [merge_to_quad(tris[i], tris[j]) for i, j in pairs]
    quads = fill_leftovers(quads, tris, adj)

    counts = [0] * len(tris)
    for quad in quads:
        corners = set(quad)
        for t, tri in enumerate(tris):
            if set(tri["v"]) <= corners:
                counts[t] += 1
    coverage = (
        sum(1 for c in counts if c == 1) / len(tris) if tris else 1.0
    )

    return {
        "pairs": pairs,
        "quads": quads,
        "triangle_count": len(tris),
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _orient2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d):
    o1 = _orient2(a, b, c)
    o2 = _orient2(a, b, d)
    o3 = _orient2(c, d, a)
    o4 = _orient2(c, d, b)
    return (o1 * o2 < -EPS) and (o3 * o4 < -EPS)


def validate_quads(quads, positions) -> list[str]:
    """Return one ``"CODE: detail"`` string per defect found; ``[]`` if valid.

    Each quad is a 4-tuple of vertex indices into ``positions``.  Per quad the
    checks run in this order:

    ``INDEX_OUT_OF_RANGE``
        an index is negative or ``>= len(positions)``.
    ``DUPLICATE_VERTEX``
        a vertex occurs twice, apart from the single allowed repeat of the
        first index at the end of a degenerate ``(a, b, c, a)`` quad.
    ``DEGENERATE_AREA``
        the Newell area is ``<= 1e-9`` (coincident or collinear corners).
    ``SELF_INTERSECTION``
        two non-adjacent boundary edges cross, i.e. a bow-tie quad; checked in
        the plane of the largest Newell-normal component, with zero-length
        edges skipped.
    ``INCONSISTENT_WINDING``
        the halves ``(v0, v1, v2)`` and ``(v0, v2, v3)`` face opposite ways
        (``dot(n_left, n_right) <= 0``); degenerate quads are skipped.

    A valid counter-clockwise quad returns ``[]``.  A globally flipped winding
    cannot be recovered from ``positions`` alone, so only the consistency of
    the two halves is reported here; the absolute winding is checked against
    the source triangle through :func:`merge_to_quad`.

    Conventions: quad winding is counter-clockwise, tolerance 1e-9.
    """
    messages = []
    n = len(positions)

    for qi, quad in enumerate(quads):
        # INDEX_OUT_OF_RANGE -- nothing else can be computed without points.
        if any(i < 0 or i >= n for i in quad):
            messages.append(
                f"INDEX_OUT_OF_RANGE: quad {qi} references a vertex outside "
                f"[0, {n})"
            )
            continue

        pts = [positions[i] for i in quad]
        degenerate = quad[0] == quad[3] and len(set(quad)) == 3

        # DUPLICATE_VERTEX
        if not degenerate and len(set(quad)) != 4:
            messages.append(
                f"DUPLICATE_VERTEX: quad {qi} repeats a vertex index"
            )

        # DEGENERATE_AREA
        newell = _newell(pts)
        area = 0.5 * _length(newell)
        if area <= EPS:
            messages.append(
                f"DEGENERATE_AREA: quad {qi} has Newell area {area}"
            )

        # SELF_INTERSECTION, in the plane of the largest normal component.
        axis = max(range(3), key=lambda k: abs(newell[k]))
        flat = [tuple(p[k] for k in range(3) if k != axis) for p in pts]
        edges = [(flat[k], flat[(k + 1) % 4]) for k in range(4)]

        def zero_length(e):
            dx = e[1][0] - e[0][0]
            dy = e[1][1] - e[0][1]
            return dx * dx + dy * dy <= EPS * EPS

        for e1, e2 in ((0, 2), (1, 3)):
            if zero_length(edges[e1]) or zero_length(edges[e2]):
                continue
            a, b = edges[e1]
            c, d = edges[e2]
            if _segments_cross(a, b, c, d):
                messages.append(
                    f"SELF_INTERSECTION: quad {qi} edges {e1} and {e2} cross"
                )
                break

        # INCONSISTENT_WINDING (degenerate quads are skipped).
        if not degenerate:
            n_left = _cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))
            n_right = _cross(_sub(pts[2], pts[0]), _sub(pts[3], pts[0]))
            if _dot(n_left, n_right) <= 0.0:
                messages.append(
                    f"INCONSISTENT_WINDING: quad {qi} halves face opposite "
                    f"ways"
                )

    return messages


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def remesh_batch(meshes, config) -> dict:
    """Remesh several meshes, each one with its own adjacency and state.

    ``meshes`` is an iterable of mesh mappings as accepted by
    :func:`preserve_seams` (``"tris"``, ``"positions"``, optional ``"seams"``,
    optional ``"threshold"``).  ``config`` may carry a global ``"threshold"``
    used when a mesh does not define its own; the default is ``0.5``.

    Every mesh is processed independently: a fresh :func:`build_adjacency` is
    built for it, and no edge key, incidence tuple, pairing, seam set or quad
    list is shared between meshes.  Reusing one adjacency table across meshes
    corrupts the next mesh, because two meshes routinely reuse the same vertex
    indices for completely different geometry.

    Return ``{"meshes": [result, ...]}`` with one :func:`preserve_seams`
    result per input mesh, in input order.  Each result must be identical to
    remeshing that mesh on its own with the same threshold, and results must
    keep their own ``triangle_count``.

    Conventions: batch state is independent per mesh, tolerance 1e-9.
    """
    config = config or {}
    results = []
    for mesh in meshes:
        local = dict(mesh)
        local["threshold"] = mesh.get(
            "threshold", config.get("threshold", 0.5)
        )
        results.append(preserve_seams(local, mesh.get("seams", [])))
    return {"meshes": results}
