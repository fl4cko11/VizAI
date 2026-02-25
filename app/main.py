# app/main.py
import asyncio
import logging

from aiogram import Bot, Dispatcher

from app.api.agent import router
from app.api.middleware import AgentMiddleware
from app.core.config import Settings


async def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = Settings()

    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.middleware(AgentMiddleware(settings))

    dp.include_router(router)

    # Удаляем pending updates (например, после падения бота)
    await bot.delete_webhook(drop_pending_updates=True)

    # Запуск бота
    logging.info("Бот запущен и готов к работе.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
