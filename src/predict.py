from __future__ import annotations

from typing import Any


def predict_from_geojson(geojson: dict[str, Any], year: int) -> dict[str, Any]:
    if not isinstance(geojson, dict):
        raise TypeError("geojson must be a dictionary")
    if not isinstance(year, int):
        raise TypeError("year must be an integer")

    return {
        "risk_score": 0.0,
        "decline_year": year,
        "status": "stub",
    }
