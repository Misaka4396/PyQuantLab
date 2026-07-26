"""PyInstaller runtime hook — pre-register streamlit.version to avoid importlib.metadata errors."""

import sys
import types

if getattr(sys, 'frozen', False):
    sv = types.ModuleType('streamlit.version')

    # Version strings
    sv.STREAMLIT_VERSION_STRING = "1.22.0"
    sv._version = "1.22.0"
    sv.__version__ = "1.22.0"

    # Function: get installed version (avoid importlib.metadata)
    def _get_installed_streamlit_version():
        return "1.22.0"

    sv._get_installed_streamlit_version = _get_installed_streamlit_version

    # Function: get latest version from PyPI (disabled — no network needed)
    def _get_latest_streamlit_version(timeout=None):
        return None

    sv._get_latest_streamlit_version = _get_latest_streamlit_version

    # Function: should show new version notice
    def should_show_new_version_notice():
        return False

    sv.should_show_new_version_notice = should_show_new_version_notice

    # Other constants
    sv.CHECK_PYPI_PROBABILITY = 0.0
    sv.PYPI_STREAMLIT_URL = "https://pypi.org/pypi/streamlit/json"

    sys.modules['streamlit.version'] = sv
