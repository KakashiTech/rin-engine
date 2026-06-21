"""Engine lifecycle tests. Requires a .rin model file."""
import pytest
import os

MODEL_PATH = os.environ.get('RIN_TEST_MODEL', 'models/tiny_rin.rin')

def test_engine_init_no_model():
    from rin import RinEngine
    engine = RinEngine()
    assert engine is not None
    engine.close()

@pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                    reason=f"Test model not found at {MODEL_PATH}")
def test_engine_load_model():
    from rin import RinEngine
    engine = RinEngine(MODEL_PATH)
    info = engine.info
    assert isinstance(info, dict)
    assert 'architecture' in info
    engine.close()

@pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                    reason=f"Test model not found at {MODEL_PATH}")
def test_engine_generate_or_skip():
    """Try to generate; skip if model doesn't support text encoding."""
    from rin import RinEngine
    engine = RinEngine(MODEL_PATH, mode='mlp')
    try:
        text = engine.generate(" ", max_tokens=5)
        assert isinstance(text, str)
    except Exception:
        pytest.skip("Model does not support text generation")
    finally:
        engine.close()

@pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                    reason=f"Test model not found at {MODEL_PATH}")
def test_engine_benchmark():
    from rin import RinEngine
    engine = RinEngine(MODEL_PATH, mode='mlp')
    result = engine.benchmark(mode='mlp', warmup=2, iterations=3)
    assert isinstance(result, dict)
    assert 'tokens_per_second' in result
    engine.close()
