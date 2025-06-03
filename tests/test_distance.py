import importlib.util
import sys
import types
from pathlib import Path
import pytest

def load_app():
    # Create minimal stub modules so app.py can be imported without heavy deps
    if 'streamlit' not in sys.modules:
        st = types.ModuleType('streamlit')
        st.set_page_config = lambda *a, **k: None
        st.write = lambda *a, **k: None
        st.progress = lambda *a, **k: types.SimpleNamespace(progress=lambda *a, **k: None)
        st.warning = lambda *a, **k: None
        st.error = lambda *a, **k: None
        st.info = lambda *a, **k: None
        st.session_state = types.SimpleNamespace()
        st.selectbox = lambda *a, **k: None
        st.checkbox = lambda *a, **k: None
        st.multiselect = lambda *a, **k: []
        st.tabs = lambda labels: [None for _ in labels]
        st.subheader = lambda *a, **k: None
        st.pyplot = lambda *a, **k: None
        st.dataframe = lambda *a, **k: None
        st.download_button = lambda *a, **k: None
        sys.modules['streamlit'] = st
    # Additional stub modules
    for name in [
        'pandas', 'requests', 'matplotlib', 'matplotlib.pyplot',
        'folium', 'streamlit_folium'
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if 'numpy' not in sys.modules:
        np = types.ModuleType('numpy')
        np.isscalar = lambda x: isinstance(x, (int, float, complex))
        sys.modules['numpy'] = np
    spec = importlib.util.spec_from_file_location('app', Path(__file__).resolve().parents[1] / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_calculate_distance():
    app = load_app()
    distance_nm = app.calculate_distance(0, 0, 0, 1)
    assert distance_nm == pytest.approx(60.04, rel=0.01)
