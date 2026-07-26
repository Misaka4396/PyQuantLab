"""PyQuantLab Desktop Launcher — PyQt5 native desktop application."""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def main():
    from ui.main_window import run_app
    run_app()


if __name__ == "__main__":
    main()
