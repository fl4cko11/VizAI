import logging
from logging.handlers import RotatingFileHandler

from app.core.config import Settings


def get_logger(settings: Settings):
    # Создаем логгер с именем 'ingestion'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Проверяем, не добавлены ли уже обработчики (чтобы избежать дублирования)
    if not logger.handlers:
        # Ротация: максимум 5 файлов по 10 МБ каждый
        handler = RotatingFileHandler(
            settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
