"""Simplify GeoJSON polygon geometry using the Ramer-Douglas-Peucker algorithm.

Reduces vertex count while preserving the overall shape, making large/detailed
polygons suitable for API footprint queries (e.g. Copernicus CDSE OData).

Usage::

    python simplify_geojson.py <input.geojson> [options]

Examples::

    # Default tolerance (0.01 degrees ≈ 1 km)
    python simplify_geojson.py resources/bo_nina_roi.geojson

    # Custom tolerance, explicit output path
    python simplify_geojson.py resources/bo_nina_roi.geojson \\
        --tolerance 0.05 \\
        --output    resources/bo_nina_roi_simplified.geojson

    # Preview point counts without writing
    python simplify_geojson.py resources/bo_nina_roi.geojson --dry-run
"""

import argparse
import json
import math
import sys
from pathlib import Path


# ── Ramer-Douglas-Peucker ────────────────────────────────────────────────────


def _point_to_segment_distance(p, a, b):
    """Perpendicular distance from point p to segment a→b."""
    ax, ay = a
    bx, by = b
    px, py = p

    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rdp(coords, tolerance):
    """Ramer-Douglas-Peucker recursion. Returns simplified coordinate list."""
    if len(coords) < 3:
        return coords

    start, end = coords[0], coords[-1]
    max_dist, max_idx = 0.0, 0

    for i in range(1, len(coords) - 1):
        d = _point_to_segment_distance(coords[i], start, end)
        if d > max_dist:
            max_dist, max_idx = d, i

    if max_dist > tolerance:
        left = _rdp(coords[: max_idx + 1], tolerance)
        right = _rdp(coords[max_idx:], tolerance)
        return left[:-1] + right

    return [start, end]


def _simplify_ring(ring, tolerance):
    """Simplify a coordinate ring, preserving closure."""
    if len(ring) < 4:
        return ring

    # Close the ring before simplification, reattach after
    open_ring = ring[:-1]
    simplified = _rdp(open_ring, tolerance)

    # RDP may reduce to just 2 points; GeoJSON rings need at least 4
    if len(simplified) < 3:
        return ring  # keep original if too much was removed

    # Re-close
    simplified.append(simplified[0])
    return simplified


# ── GeoJSON traversal ────────────────────────────────────────────────────────


def _simplify_geometry(geom, tolerance):
    """Recursively simplify a GeoJSON geometry object in-place."""
    gtype = geom["type"]

    if gtype == "Polygon":
        geom["coordinates"] = [
            _simplify_ring(ring, tolerance) for ring in geom["coordinates"]
        ]

    elif gtype == "MultiPolygon":
        geom["coordinates"] = [
            [_simplify_ring(ring, tolerance) for ring in polygon]
            for polygon in geom["coordinates"]
        ]

    elif gtype == "GeometryCollection":
        for g in geom.get("geometries", []):
            _simplify_geometry(g, tolerance)

    # Point, MultiPoint, LineString, MultiLineString — no simplification needed
    return geom


def _walk_and_simplify(obj, tolerance):
    """Walk a GeoJSON object and simplify all polygon geometries."""
    t = obj["type"]

    if t == "FeatureCollection":
        for feature in obj.get("features", []):
            _walk_and_simplify(feature, tolerance)

    elif t == "Feature":
        if obj.get("geometry"):
            _simplify_geometry(obj["geometry"], tolerance)

    else:  # bare geometry
        _simplify_geometry(obj, tolerance)


# ── Counting ─────────────────────────────────────────────────────────────────


def _count_polygon_points(obj):
    """Return total vertex count across all polygon rings in the GeoJSON."""
    total = 0
    t = obj.get("type")

    if t == "FeatureCollection":
        for f in obj.get("features", []):
            total += _count_polygon_points(f)

    elif t == "Feature":
        if obj.get("geometry"):
            total += _count_polygon_points(obj["geometry"])

    elif t == "Polygon":
        total += sum(len(ring) for ring in obj.get("coordinates", []))

    elif t == "MultiPolygon":
        total += sum(
            len(ring) for polygon in obj.get("coordinates", []) for ring in polygon
        )

    elif t == "GeometryCollection":
        for g in obj.get("geometries", []):
            total += _count_polygon_points(g)

    return total


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser():
    p = argparse.ArgumentParser(
        description="Simplify GeoJSON polygon geometry (Ramer-Douglas-Peucker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", metavar="INPUT", help="Input GeoJSON file.")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        metavar="DEG",
        help=(
            "Simplification tolerance in decimal degrees. "
            "0.01° ≈ 1 km. Increase to reduce point count further."
        ),
    )
    p.add_argument(
        "--output",
        metavar="OUTPUT",
        default=None,
        help=(
            "Output file path. Defaults to <input-stem>_simplified.geojson "
            "in the same directory."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print point counts and exit without writing any file.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    with open(input_path, encoding="utf-8") as fh:
        geojson = json.load(fh)

    points_before = _count_polygon_points(geojson)

    if args.dry_run:
        import copy

        preview = copy.deepcopy(geojson)
        _walk_and_simplify(preview, args.tolerance)
        points_after = _count_polygon_points(preview)
        reduction = (1 - points_after / points_before) * 100 if points_before else 0
        print(f"Input  : {input_path}")
        print(f"Tolerance       : {args.tolerance}°")
        print(f"Points before   : {points_before:,}")
        print(f"Points after    : {points_after:,}")
        print(f"Reduction       : {reduction:.1f}%")
        print("(dry run — no file written)")
        return 0

    _walk_and_simplify(geojson, args.tolerance)
    points_after = _count_polygon_points(geojson)
    reduction = (1 - points_after / points_before) * 100 if points_before else 0

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            f"{input_path.stem}_simplified{input_path.suffix}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, separators=(",", ":"))

    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    print(f"Tolerance       : {args.tolerance}°")
    print(f"Points before   : {points_before:,}")
    print(f"Points after    : {points_after:,}")
    print(f"Reduction       : {reduction:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
