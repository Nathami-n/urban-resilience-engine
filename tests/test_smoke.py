from api.main import app
from src.predict import predict_from_geojson


def test_predict_stub_shape() -> None:
    result = predict_from_geojson({"type": "FeatureCollection", "features": []}, 2025)

    assert "risk_score" in result
    assert "decline_year" in result


def test_api_title() -> None:
    assert app.title == "Urban Resilience Engine"