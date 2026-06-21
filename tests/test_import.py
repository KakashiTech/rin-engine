"""Test that the rin package imports correctly."""

def test_import_rin():
    import rin
    assert hasattr(rin, 'RinEngine')
    assert hasattr(rin, 'RinException')
    assert hasattr(rin, 'MODE_MLP')
    assert hasattr(rin, 'MODE_TRANSFORMER')
    assert hasattr(rin, 'HAS_NATIVE')

def test_rin_engine_class():
    from rin import RinEngine
    assert RinEngine is not None
    # Version should be callable without instance
    v = RinEngine.version()
    assert isinstance(v, str)
    assert len(v) > 0

def test_backend_exists():
    from rin._backend import _backend_name
    name = _backend_name()
    assert isinstance(name, str) and len(name) > 0
