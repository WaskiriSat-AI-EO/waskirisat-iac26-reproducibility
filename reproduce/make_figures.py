"""Generate Fig. 1 and Fig. 2 as single-column (85 mm) grayscale SVGs. No dependencies."""

import html

FONT = "Liberation Serif, Times New Roman, serif"
FS = 7.0
LEAD = 8.4


def box(x, y, w, h, lines, fill="#ffffff", bold=0, stroke=True):
    s = ""
    if stroke:
        s += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#000" stroke-width="0.6"/>'
    n = len(lines)
    cy = y + h / 2 - (n - 1) * LEAD / 2 + FS * 0.35
    for i, line in enumerate(lines):
        weight = "bold" if i < bold else "normal"
        s += (
            f'<text x="{x + w / 2:.2f}" y="{cy + i * LEAD:.2f}" font-size="{FS}" '
            f'font-weight="{weight}" text-anchor="middle">{html.escape(line)}</text>'
        )
    return s


def arrow(x1, y1, x2, y2):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2 - 0.5}" stroke="#000" '
        f'stroke-width="0.6" marker-end="url(#a)"/>'
    )


def svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
        '<defs><marker id="a" markerWidth="5" markerHeight="5" refX="4.5" refY="2.5" '
        'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L5,2.5 L0,5 z"/></marker></defs>'
        f'<rect width="{w}" height="{h}" fill="#fff"/>{body}</svg>'
    )


def fig1():
    b = ""
    b += box(
        4,
        4,
        114,
        36,
        ["Visible imager", "PixArt PAG7936LT", "1280 × 800 px, global shutter"],
        "#e8e8e8",
        1,
    )
    b += box(
        123,
        4,
        114,
        36,
        ["Thermal imager", "FLIR Lepton 3.5 microbolometer", "160 × 120 px, 8–14 µm"],
        "#e8e8e8",
        1,
    )
    b += arrow(61, 40, 61, 52) + arrow(180, 40, 180, 52)
    b += '<rect x="4" y="52" width="233" height="112" fill="#f6f6f6" stroke="#000" stroke-width="0.6"/>'
    b += box(
        4, 55, 233, 12, ["Processing core: OpenMV N6 (STM32N657)"], bold=1, stroke=False
    )
    b += box(10, 70, 107, 24, ["Arm Cortex-M55", "800 MHz, Helium"], "#ffffff", 1)
    b += box(
        124,
        70,
        107,
        24,
        ["Neural-ART accelerator", "600 GOPS, 8-bit integer"],
        "#ffffff",
        1,
    )
    b += box(
        10,
        102,
        107,
        28,
        ["Cloud (IQC) classifier", "MobileNetV2, 224 × 224 × 3"],
        "#dcdcdc",
        1,
    )
    b += box(
        124,
        102,
        107,
        28,
        ["Thermal classifier", "MobileNetV3-Small, 160 × 120"],
        "#dcdcdc",
        1,
    )
    b += box(
        4,
        136,
        233,
        24,
        [
            "4.2 MB SRAM · 64 MB SDRAM · 32 MB flash (artifacts ≤ 2 MB)",
            "Frame storage: microSD",
        ],
        stroke=False,
    )
    b += arrow(120.5, 164, 120.5, 176)
    b += box(
        4,
        176,
        233,
        26,
        [
            "Bus interface: 50-pin PC/104-style connector",
            "power · command and telemetry · frame transfer",
        ],
        "#e8e8e8",
        1,
    )
    b += arrow(61, 202, 61, 214) + arrow(180, 202, 180, 214)
    b += box(
        4,
        214,
        114,
        36,
        [
            "UHF downlink queue",
            "non-obscured visible frames;",
            "thermal frame on request",
        ],
        "#ffffff",
        1,
    )
    b += box(
        123,
        214,
        114,
        36,
        ["Telemetry beacon", "anomaly flag on a", "thermal positive"],
        "#ffffff",
        1,
    )
    return svg(241, 254, b)


def fig2():
    b = ""
    b += box(66, 4, 83, 14, ["Thermal channel"], bold=1, stroke=False)
    b += box(154, 4, 83, 14, ["Visible channel"], bold=1, stroke=False)
    rh, gap = 42, 10
    rows = [
        (
            ["1  Acquisition"],
            ["Sentinel-3 SLSTR L1B", "F2 channel, 10.85 µm", "1 km nadir"],
            ["Sentinel-2 MSI L2A", "1280 × 800 px tiles", "10 m (12.8 × 8 km)"],
        ),
        (
            ["2  Label", "derivation"],
            [
                "Summed FIRMS FRP, ±1 d",
                "≥ 20,000 MW: fire",
                "no detection: no fire",
                "ambiguous: excluded",
            ],
            [
                "Scene classification",
                "cloud > 0.70, clear < 0.30",
                "intermediate excluded",
            ],
        ),
        (
            ["3  Detector", "simulation"],
            [
                "Planck 4 × 4 mean (4 km),",
                "broadband corr., noise,",
                "fixed 8-bit window,",
                "PSF, FPN",
            ],
            ["PSF (≈56× GSD gap),", "shot and read noise,", "8-bit, FPN, JPEG"],
        ),
    ]
    y = 22
    for stage, t_, v in rows:
        b += box(4, y, 58, rh, stage, "#e8e8e8", len(stage))
        b += box(66, y, 83, rh, t_)
        b += box(154, y, 83, rh, v)
        b += arrow(107.5, y + rh, 107.5, y + rh + gap) + arrow(
            195.5, y + rh, 195.5, y + rh + gap
        )
        y += rh + gap
    b += box(4, y, 58, rh, ["4  Degradation", "profile"], "#e8e8e8", 2)
    b += box(
        66,
        y,
        171,
        rh,
        [
            "A: best-case control (withheld)",
            "B: mid-cycle operation (training)",
            "C: stress (withheld)",
        ],
    )
    b += arrow(151.5, y + rh, 151.5, y + rh + gap)
    y += rh + gap
    b += box(4, y, 58, rh, ["5  Partitioning"], "#e8e8e8", 1)
    b += box(
        66,
        y,
        171,
        rh,
        [
            "Granule-level hash assignment,",
            "precedence test > validation > training;",
            "class balancing on training only",
        ],
    )
    return svg(241, y + rh + 4, b)


for name, fn in [
    ("fig1_payload_block_diagram", fig1),
    ("fig2_synthesis_pipeline", fig2),
]:
    with open(f"{name}.svg", "w", encoding="utf-8") as f:
        f.write(fn())
    print("wrote", name)
