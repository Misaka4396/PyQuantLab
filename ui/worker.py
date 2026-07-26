"""Background worker threads for non-blocking UI operations."""

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class Worker(QObject):
    """QObject worker that runs a callable in a background thread."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def run_in_thread(parent, func, *args, on_finished=None, on_error=None, **kwargs):
    """Launch *func* in a QThread, return the thread so caller can track it.

    The thread is auto-deleted after finish/error.
    """
    thread = QThread(parent)
    worker = Worker(func, *args, **kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)

    if on_finished is not None:
        worker.finished.connect(on_finished)
    if on_error is not None:
        worker.error.connect(on_error)

    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(worker.deleteLater)

    thread.start()
    return thread
