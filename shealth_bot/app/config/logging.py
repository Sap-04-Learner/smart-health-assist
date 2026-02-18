import logging.config
import os


def setup_logging(settings):
    os.makedirs(settings.logs_path, exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(settings.logs_path, "health_assist.log"),
                "maxBytes": 10_485_760,  # 10MB
                "backupCount": 5,
                "formatter": "standard",
                "level": settings.log_level.upper(),
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": settings.log_level.upper(),
            },
        },
        "root": {
            "level": settings.log_level.upper(),
            "handlers": ["file", "console"],
        },
    }

    logging.config.dictConfig(logging_config)