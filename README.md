# 🏭 SuperApp - Industrial AI Trainer

Система для обучения и анализа промышленных языковых моделей на основе технической документации.

## 🎯 Назначение

Этот проект предназначен для создания специализированных AI-моделей, способных анализировать техническую документацию промышленного оборудования, отвечать на вопросы о технологических процессах и помогать инженерам в решении технических задач.

## 🏗️ Архитектура проекта

```
superapp/
├── 📁 app/              # Основные скрипты
│   ├── pdf_to_qa.py     # Конвертация PDF в Q&A
│   ├── fix_errors.py    # Исправление ошибок в данных
│   ├── merge_datasets.py # Объединение датасетов
│   ├── train_qlora.py   # QLoRA обучение
│   ├── continue_training.py # Продолжение обучения
│   ├── test_trained_model.py # Тестирование модели
│   └── compare_models.py # Сравнение моделей
├── 📁 input/            # Входные данные и датасеты
├── 📁 output/           # Результаты обработки
├── 📁 models/           # Базовые модели
├── 📁 trained-models/  # Обученные модели
├── 📁 log/              # Логи работы
├── requirements.txt     # Зависимости Python
└── README.md           # Документация
```

## 🚀 Быстрый старт

### 1. Клонирование и установка
```bash
git clone <repository-url>
cd superapp
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Подготовка данных
Создайте JSON файл с датасетом в формате:
```json
{
  "qa_pairs": [
    {
      "question": "Как работает шаровая мельница?",
      "answer": "Шаровая мельница работает за счет вращения барабана..."
    }
  ]
}
```

### 3. Обучение модели
```bash
python app/train_qlora.py --data input/dataset.json --base-model /path/to/base/model --output-dir trained-models/model_v1
```

### 4. Тестирование модели
```bash
python app/compare_models.py --trained-model trained-models/model_v1
```

## 🎯 Основные возможности

### 🎓 Обучение моделей
- **QLoRA обучение** с поддержкой последовательного обучения
- **Первичное обучение** на новых доменах знаний
- **Продолжение обучения** на дополнительных датасетах
- **Адаптация** под различные технические области

### 🔍 Анализ и тестирование
- **Сравнение моделей** до и после обучения
- **Оценка качества** ответов на технические вопросы
- **Мониторинг** процесса обучения
- **Валидация** результатов

### 📊 Обработка данных
- **Подготовка датасетов** в формате вопрос-ответ
- **Токенизация** и предобработка текста
- **Форматирование** для обучения языковых моделей

## 📋 Примеры использования

### Конвертация PDF в Q&A
```bash
python app/pdf_to_qa.py input/document.pdf -o output/dataset.json -m qwen2.5:3b -q 5
```

### Обучение модели
```bash
python app/train_qlora.py \
    --data output/dataset.json \
    --base-model /path/to/base/model \
    --output-dir trained-models/model_v1 \
    --epochs 3 \
    --learning-rate 2e-4
```

### Продолжение обучения
```bash
python app/continue_training.py \
    --data input/new_data.json \
    --base-model /path/to/base/model \
    --trained-model trained-models/model_v1 \
    --output-dir trained-models/model_v2
```

### Сравнение моделей
```bash
python app/compare_models.py \
    --base-model /path/to/base/model \
    --trained-model trained-models/model_v1 \
    --questions "Как работает насос?" "Что такое диспергатор?"
```

## ⚙️ Требования

- **Python** 3.8+
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **PEFT** (Parameter-Efficient Fine-Tuning)
- **GPU** с поддержкой CUDA (рекомендуется)
- **RAM** минимум 16GB

## 🔧 Конфигурация

### Модели
Поддерживаемые базовые модели:
- **Llama-2** (7B, 13B, 70B)
- **Mistral** (7B, 8x7B)
- **Qwen** (2.5-3B, 7B, 14B)
- **CodeLlama** (7B, 13B, 34B)
- **Yandex** модели

### Параметры обучения
- **LoRA Rank**: 8-64 (по умолчанию 16)
- **Learning Rate**: 1e-5 до 2e-4
- **Batch Size**: 1-16 (зависит от GPU)
- **Epochs**: 1-10 (зависит от задачи)

## 📊 Мониторинг и логирование

- **Консольные логи** в реальном времени
- **Автосохранение** моделей каждые 100 шагов
- **Метрики качества** на каждом этапе

## 🚨 Устранение неполадок

### Частые проблемы
1. **Недостаточно памяти** - уменьшите batch-size
2. **Медленное обучение** - проверьте использование GPU
3. **Ошибки загрузки** - проверьте доступ к Hugging Face

### Поддержка
- Проверьте логи в консоли
- Убедитесь в корректности путей к файлам
- Проверьте совместимость версий библиотек

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT.

## 🙏 Благодарности

- Hugging Face за библиотеки Transformers и PEFT
- Microsoft за LoRA метод
- Сообщество open-source за вклад в развитие

---

**SuperApp** - Создаем будущее промышленной аналитики с помощью AI! 🚀
