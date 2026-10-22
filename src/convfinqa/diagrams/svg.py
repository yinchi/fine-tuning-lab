"""A tiny SVG builder: boxes, text, and routed arrows with rounded corners. No dependencies.

Coordinates are SVG coordinates: x grows to the right and y grows *downward*.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from xml.sax.saxutils import escape

_DIRECTIONS = {"r": (1.0, 0.0), "l": (-1.0, 0.0), "u": (0.0, -1.0), "d": (0.0, 1.0)}
_MOVE = re.compile(r"([rlud])(\d+(?:\.\d+)?)")

Point = tuple[float, float]


def waypoints(start: Point, moves: str) -> list[Point]:
    """Turn ``"r20 u100 l20"`` (right 20, up 100, left 20) into the points of a path starting at ``start``."""
    tokens = moves.split()
    parsed = [_MOVE.fullmatch(t) for t in tokens]
    if not tokens or any(m is None for m in parsed):
        raise ValueError(f"bad moves {moves!r}: expected tokens like 'r20 u100 l5.5'")
    points = [start]
    for m in parsed:
        dx, dy = _DIRECTIONS[m.group(1)]
        length = float(m.group(2))
        x, y = points[-1]
        points.append((x + dx * length, y + dy * length))
    return points


def _unit(a: Point, b: Point) -> Point:
    length = math.dist(a, b)
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def rounded_path(points: list[Point], radius: float) -> str:
    """An SVG path through ``points`` with each interior corner rounded.

    The radius shrinks at a corner when a neighbouring segment is too short to hold it (each segment gives
    at most half its length to each end), so short segments never produce kinks.
    """
    d = [f"M {points[0][0]:g} {points[0][1]:g}"]
    for i in range(1, len(points) - 1):
        prev, corner, nxt = points[i - 1], points[i], points[i + 1]
        r = min(radius, math.dist(prev, corner) / 2, math.dist(corner, nxt) / 2)
        u_in, u_out = _unit(prev, corner), _unit(corner, nxt)
        a = (corner[0] - u_in[0] * r, corner[1] - u_in[1] * r)
        b = (corner[0] + u_out[0] * r, corner[1] + u_out[1] * r)
        d.append(f"L {a[0]:.2f} {a[1]:.2f} Q {corner[0]:g} {corner[1]:g} {b[0]:.2f} {b[1]:.2f}")
    d.append(f"L {points[-1][0]:g} {points[-1][1]:g}")
    return " ".join(d)


def arrowhead(tip: Point, direction: Point, length: float = 9.0, half_width: float = 4.5) -> list[Point]:
    """The three corners of an arrowhead whose point is at ``tip`` and which points along ``direction``."""
    ux, uy = direction
    bx, by = tip[0] - ux * length, tip[1] - uy * length
    return [tip, (bx - uy * half_width, by + ux * half_width), (bx + uy * half_width, by - ux * half_width)]


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def top(self) -> Point:
        return (self.x + self.w / 2, self.y)

    @property
    def bottom(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def left(self) -> Point:
        return (self.x, self.y + self.h / 2)

    @property
    def right(self) -> Point:
        return (self.x + self.w, self.y + self.h / 2)


class Svg:
    FONT = "Helvetica, Arial, sans-serif"
    INK = "#2b2b2b"

    def __init__(self, width: float, height: float):
        self.width, self.height = width, height
        self.items: list[str] = [f'<rect width="{width:g}" height="{height:g}" fill="white"/>']

    def box(self, box: Box, fill: str, stroke: str, radius: float = 8, dashed: bool = False) -> None:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.items.append(
            f'<rect x="{box.x:g}" y="{box.y:g}" width="{box.w:g}" height="{box.h:g}" rx="{radius:g}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )

    def circle(self, centre: Point, radius: float, fill: str = "white", stroke: str | None = None) -> None:
        stroke = stroke or self.INK
        self.items.append(
            f'<circle cx="{centre[0]:g}" cy="{centre[1]:g}" r="{radius:g}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )

    def text(self, anchor: Point, lines: list[str] | str, size: float = 13, bold: bool = False,
             align: str = "middle", colour: str | None = None, line_height: float | None = None) -> None:
        """Text with its first line's baseline at ``anchor``; extra lines go below it."""
        if isinstance(lines, str):
            lines = [lines]
        step = line_height or size * 1.3
        weight = ' font-weight="bold"' if bold else ""
        spans = "".join(
            f'<tspan x="{anchor[0]:g}" dy="{0 if i == 0 else step:g}">{escape(line)}</tspan>' for i, line in enumerate(lines)
        )
        self.items.append(
            f'<text x="{anchor[0]:g}" y="{anchor[1]:g}" font-family="{self.FONT}" font-size="{size:g}"{weight} '
            f'text-anchor="{align}" fill="{colour or self.INK}">{spans}</text>'
        )

    def dot(self, centre: Point, radius: float = 3) -> None:
        self.items.append(f'<circle cx="{centre[0]:g}" cy="{centre[1]:g}" r="{radius:g}" fill="{self.INK}"/>')

    def route(self, start: Point, moves: str, radius: float = 10, arrow: bool = True, dashed: bool = False,
              colour: str | None = None) -> Point:
        """Draw an arrow along ``moves`` (see :func:`waypoints`) with rounded corners. Returns the end point."""
        colour = colour or self.INK
        points = waypoints(start, moves)
        tip = points[-1]
        direction = _unit(points[-2], tip)
        line = list(points)
        if arrow:  # stop the line at the arrowhead's base so the line does not poke through its point
            length = min(9.0, math.dist(points[-2], tip))
            line[-1] = (tip[0] - direction[0] * length, tip[1] - direction[1] * length)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.items.append(
            f'<path d="{rounded_path(line, radius)}" fill="none" stroke="{colour}" stroke-width="1.5" '
            f'stroke-linejoin="round"{dash}/>'
        )
        if arrow:
            head = " ".join(f"{x:.2f},{y:.2f}" for x, y in arrowhead(tip, direction))
            self.items.append(f'<polygon points="{head}" fill="{colour}"/>')
        return tip

    def to_string(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:g}" height="{self.height:g}" '
            f'viewBox="0 0 {self.width:g} {self.height:g}">\n' + "\n".join(self.items) + "\n</svg>\n"
        )
