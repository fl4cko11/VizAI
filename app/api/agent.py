# app/api/agent.py
import os
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, Message

from app.schemas.agent_state import (
    AgentState,
    DatasetProfile,
    TelegramContext,
)
from app.services.parse import (
    generate_dataset_profile,  # ← Предполагаем, что у тебя есть такая функция
)

router = Router()


# --- FSM States ---
class AgentStates(StatesGroup):
    wait_for_csv = State()  # Ожидание CSV файла
    wait_for_human = State()  # Ожидание корректировок от пользователя


# --- /start ---
@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    await message.answer(
        "👋 Привет! Я — ИИ-агент для визуализации данных.\n\n"
        "Загрузи CSV-файл, и я помогу построить графики, диаграммы и всё, что нужно.\n\n"
        "Пришли файл:"
    )
    await state.set_state(AgentStates.wait_for_csv)


# --- Хэндлер: ожидание CSV файла ---
@router.message(AgentStates.wait_for_csv, F.document.mime_type == "text/csv")
async def handle_csv(message: Message, state: FSMContext, agent: Any):
    # Скачивание файла
    file_id = message.document.file_id
    file = await message.bot.get_file(file_id)
    file_path = f"app/temp_csv/{file_id}.csv"

    os.makedirs("app/temp_csv", exist_ok=True)
    await message.bot.download_file(file.file_path, destination=file_path)

    try:
        dataset_profile: DatasetProfile = generate_dataset_profile(file_path, file_id)
    except Exception as e:
        await message.answer(
            f"❌ Ошибка при обработке файла: {str(e)}\nПопробуй другой."
        )
        return

    user_query = (message.caption or "Проанализируй и визуализируй эти данные.").strip()

    # Подготовка initial_state
    telegram_context = TelegramContext(
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        username=message.from_user.username,
    )

    initial_state = AgentState(
        user_query=user_query,
        user_data=[dataset_profile],
        chat_history=[
            {
                "role": "user",
                "content": user_query,
            }
        ],
        artifacts=[],
        used_methods=[],
        answer=None,
        errors=[],
        iteration=0,
        telegram=telegram_context,
    )

    # Запуск агента
    try:
        result = await agent.ainvoke(initial_state)
    except Exception as e:
        await message.answer(f"❌ Ошибка при работе агента: {str(e)}")
        return

    # Отправка ответа
    answer_text = result.get("answer", "Анализ завершён.")
    await message.answer(answer_text)

    # Отправка HTML-артфактов
    artifacts = result.get("artifacts", [])
    for artifact in artifacts:
        html_file = FSInputFile(artifact.file_path, filename=artifact.file_name)
        await message.answer_document(html_file)

    # Сохраняем состояние для следующей итерации
    await state.update_data(agent_state=result)
    await state.set_state(AgentStates.wait_for_human)

    # Подсказка пользователю
    await message.answer(
        "📬 Теперь можешь попросить изменить график, добавить стиль или выбрать другой тип.\n"
        "Напиши, что хочешь изменить, или используй /end чтобы завершить."
    )


# --- Хэндлер: ожидание корректировок от пользователя ---
@router.message(AgentStates.wait_for_human, F.text, ~F.text.startswith("/"))
async def handle_correction(message: Message, state: FSMContext, agent: Any):
    user_query = message.text.strip()

    # Получаем предыдущее состояние агента
    data = await state.get_data()
    prev_state = data.get("agent_state")

    if not prev_state:
        await message.answer("❌ Сессия устарела. Начни сначала — /start")
        await state.clear()
        return

    # Обновляем состояние
    updated_state = prev_state.copy()
    updated_state["user_query"] = user_query
    updated_state["chat_history"].append({"role": "user", "content": user_query})

    # Запуск агента
    try:
        result = await agent.ainvoke(updated_state)
    except Exception as e:
        await message.answer(f"❌ Ошибка при обработке запроса: {str(e)}")
        return

    # Ответ
    answer_text = result.get("answer", "Анализ завершён.")
    await message.answer(answer_text)

    # Отправка новых артефактов
    artifacts = result.get("artifacts", [])
    for artifact in artifacts:
        html_file = FSInputFile(artifact.file_path, filename=artifact.file_name)
        await message.answer_document(html_file)

    # Сохранение обновлённого состояния
    await state.update_data(agent_state=result)
    await state.set_state(AgentStates.wait_for_human)

    # Подсказка пользователю
    await message.answer(
        "📬 Теперь можешь попросить изменить график, добавить стиль или выбрать другой тип.\n"
        "Напиши, что хочешь изменить, или используй /end чтобы завершить."
    )


# --- Хэндлер: завершение сессии ---
@router.message(Command("end"))
async def cmd_end(message: Message, state: FSMContext):
    # Очистка временных файлов
    data = await state.get_data()
    agent_state = data.get("agent_state", {})

    files_to_remove = []

    # Удаляем CSV
    for dataset in agent_state.get("user_data", []):
        csv_path = dataset.file_path
        if csv_path and os.path.exists(csv_path):
            files_to_remove.append(csv_path)

    # Удаляем HTML
    for artifact in agent_state.get("artifacts", []):
        html_path = artifact.file_path
        if html_path and os.path.exists(html_path):
            files_to_remove.append(html_path)

    # Удаление
    removed_count = 0
    for file_path in set(files_to_remove):
        try:
            os.remove(file_path)
            removed_count += 1
        except Exception as e:
            print(f"Не удалось удалить {file_path}: {e}")

    # Сброс состояния
    await state.clear()

    await message.answer(
        f"✅ Сессия завершена. Удалено {removed_count} временных файлов.\n"
        "Чтобы начать снова — отправь /start"
    )
