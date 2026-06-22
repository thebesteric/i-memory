import logging


class SafeStreamHandler(logging.StreamHandler):
    """Ignore closed-stream writes during interpreter/test teardown."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except ValueError as exc:
            if "closed file" in str(exc).lower():
                return
            raise

