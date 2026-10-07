import logging
from pathlib import Path


LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"


def get_logger(log_path: str | Path | None = None):
    logger = logging.getLogger("ga")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        file_path = Path(log_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path = str(file_path.resolve())

        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename != resolved_path:
                logger.removeHandler(handler)
                handler.close()

        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
