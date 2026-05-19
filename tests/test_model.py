from app.model import predict_text

def test_predict_text_shape():
    text = "win cash now free"
    result = predict_text(text)
    assert isinstance(result, dict)
    assert result["label"] in ("ham", "spam")
    assert 0.0 <= result["score"] <= 1.0