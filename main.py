"""
VideoFIT — Entry Point
Initialises the IC4 library, creates the QApplication, and launches the orchestrator.
"""

import sys

import imagingcontrol4 as ic4
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.orchestrator import AppOrchestrator

# ── Developer flag ────────────────────────────────────────────────────────────
# Set to True to enable the Debug Preprocessing panel in the Settings menu.
# Leave False for production / end-user builds.
DEBUG = True


def main() -> None:
    ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)

    try:
        qt_app = QApplication(sys.argv)
        qt_app.setStyle("Fusion")
        qt_app.setFont(QFont("Segoe UI", 10))

        orchestrator = AppOrchestrator(debug=DEBUG)
        orchestrator.show()

        exit_code = qt_app.exec()

        # Explicit cleanup before library teardown
        orchestrator.cleanup()
        del orchestrator  # ensure IC4 objects are GC'd before Library.exit()
    finally:
        ic4.Library.exit()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

