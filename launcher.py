"""PyQuantLab Launcher — starts Streamlit server and opens browser."""

import os
import sys
import threading
import time
import webbrowser

# Ensure the project directory is on sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def open_browser(host, port):
    """Open the browser after a short delay to let the server start."""
    time.sleep(2)
    webbrowser.open(f"http://{host}:{port}")


def main():
    host = "localhost"
    port = 8501

    print(f"Starting PyQuantLab on http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    # Start browser in background thread
    t = threading.Thread(target=open_browser, args=(host, port), daemon=True)
    t.start()

    # Run Streamlit server (runtime_hook.py patches streamlit.version before import)
    from streamlit.web.bootstrap import run

    app_path = os.path.join(PROJECT_DIR, "app.py")
    run(app_path, command_line=None, args=[], flag_options={})


if __name__ == "__main__":
    main()
