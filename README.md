# 🏭 SuperApp - RAG система для горнодобывающей документации

Интеллектуальная система поиска и генерации ответов на основе горнодобывающих учебников с использованием гибридного поиска (ключевые слова + семантический поиск).

## 🎯 Основные компоненты

### 1. **pdf_to_qa.py** - Генератор базы знаний
Скрипт для автоматического создания пар вопрос-ответ из PDF документов с использованием Ollama LLM.

### 2. **rag_system.py** - RAG система
Основная система поиска и генерации ответов с гибридным поиском и интеграцией с Ollama.

## ✨ Ключевые возможности

- 🔄 **Конвертация PDF в Q&A** - автоматическое создание датасетов из технической документации
- 🔍 **Гибридный поиск** - комбинация ключевого и семантического поиска
- 🤖 **RAG система** - интеллектуальные ответы на основе найденной информации
- 🎓 **QLoRA обучение** - эффективное fine-tuning с минимальными ресурсами (исследовательский)
- 🔄 **Продолжение обучения** - последовательное улучшение моделей на новых данных (исследовательский)
- 📊 **Сравнение моделей** - оценка качества обучения (исследовательский)
- 🛠️ **Исправления ошибок** - автоматическая обработка проблемных данных (исследовательский)
- 📈 **Мониторинг** - детальное логирование и отслеживание прогресса (исследовательский)

## 🚀 Быстрый старт

### 1. Создание базы знаний из PDF
```bash
# Создание Q&A пар из PDF документа
python app/pdf_to_qa.py input/your-document.pdf -o output/knowledge-base.json -q 20 --ollama-url https://your-ollama-server:11436 --api-token YOUR_TOKEN
```

### 2. Запуск RAG системы
```bash
# Интерактивный режим
python app/rag_system.py --mode interactive --data output/knowledge-base.json

# Демонстрационный режим
python app/rag_system.py --mode demo --data output/knowledge-base.json

# Список доступных баз знаний
python app/rag_system.py --list-data
```

### 3. Тестирование системы
```bash
# Полное тестирование
python app/test_rag.py --test all --data output/knowledge-base.json
```

## 🏗️ Архитектура проекта

```
superapp/
├── 📁 app/              # Основные скрипты
│   ├── pdf_to_qa.py     # 🎯 ОСНОВНОЙ: Генератор базы знаний
│   ├── rag_system.py    # 🎯 ОСНОВНОЙ: RAG система
│   ├── test_rag.py      # Тестирование RAG системы
│   ├── run_rag.py       # Простой запуск RAG
│   ├── start_rag.sh     # Bash скрипт для запуска
│   ├── fix_errors.py    # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: Исправление ошибок в данных
│   ├── merge_datasets.py # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: Объединение датасетов
│   ├── train_qlora.py   # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: QLoRA обучение
│   ├── continue_training.py # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: Продолжение обучения
│   ├── test_trained_model.py # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: Тестирование модели
│   └── compare_models.py # 🔬 ИССЛЕДОВАТЕЛЬСКИЙ: Сравнение моделей
├── 📁 input/            # Входные PDF документы
├── 📁 output/           # Сгенерированные базы знаний
│   ├── base-mining-and-mining-quality.final.json
│   └── book-machine-equip.json
├── 📁 models/           # Базовые модели (для исследовательских скриптов)
├── 📁 trained-models/  # Обученные модели (для исследовательских скриптов)
├── 📁 log/              # Логи работы
├── requirements.txt     # Зависимости Python
└── README.md           # Документация
```

## 📚 Детальная документация по основным компонентам

### pdf_to_qa.py - Генератор базы знаний

#### Назначение
Автоматически извлекает текст из PDF документов, разбивает на чанки и генерирует пары вопрос-ответ с помощью LLM.

#### Параметры командной строки
```bash
python app/pdf_to_qa.py [OPTIONS] INPUT_PDF

Обязательные:
  INPUT_PDF              Путь к входному PDF файлу

Опции:
  -o, --output OUTPUT    Выходной JSON файл (по умолчанию: output/generated_qa.json)
  -q, --questions N      Количество вопросов на чанк (по умолчанию: 5)
  --ollama-url URL       URL сервера Ollama (по умолчанию: http://localhost:11434)
  --api-token TOKEN      Токен доступа для Ollama API
  --model MODEL          Модель LLM (по умолчанию: llama2:7b)
  --chunk-size N         Размер чанка в символах (по умолчанию: 2000)
  --overlap N            Перекрытие между чанками (по умолчанию: 200)
  --max-retries N        Максимальное количество повторов (по умолчанию: 3)
  --timeout N            Таймаут запроса в секундах (по умолчанию: 30)
  --verbose              Подробный вывод
  --help                 Показать справку
```

#### Примеры использования
```bash
# Базовое использование
python app/pdf_to_qa.py input/mining-textbook.pdf -o output/mining_qa.json -q 10

# С настройкой Ollama сервера
python app/pdf_to_qa.py input/textbook.pdf \
  --ollama-url https://193.247.73.14:11436 \
  --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF \
  --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest \
  -o output/qa_pairs.json -q 15

# С подробным выводом
python app/pdf_to_qa.py input/book.pdf -o output/qa.json -q 20 --verbose
```

#### Выходной формат
```json
{
  "source_pdf": "filename.pdf",
  "total_questions": 150,
  "generation_date": "2025-01-02 12:00:00",
  "model_used": "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest",
  "qa_pairs": [
    {
      "question": "Что такое коэффициент крепости Протодьяконова?",
      "answer": "Коэффициент крепости Протодьяконова - это числовая характеристика..."
    }
  ]
}
```

### rag_system.py - RAG система

#### Назначение
Система поиска и генерации ответов, использующая гибридный подход (TF-IDF + семантический поиск) для поиска релевантной информации и генерации ответов с помощью LLM.

#### Параметры командной строки
```bash
python app/rag_system.py [OPTIONS]

Опции:
  --mode {interactive,demo}    Режим работы (по умолчанию: interactive)
  --model MODEL                Модель LLM (по умолчанию: yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest)
  --data PATH                  Путь к базе знаний JSON
  --knowledge-base PATH        Альтернативное имя для --data
  --list-data                  Показать доступные базы знаний
  --ollama-url URL             URL сервера Ollama (по умолчанию: https://localhost:11434)
  --api-token TOKEN            Токен доступа для Ollama API
  --max-context N              Максимальное количество контекстных документов (по умолчанию: 5)
  --similarity-threshold F     Порог схожести для семантического поиска (по умолчанию: 0.7)
  --help                       Показать справку
```

#### Режимы работы

**1. Интерактивный режим**
```bash
python app/rag_system.py --mode interactive --data output/knowledge-base.json
```
- Позволяет задавать вопросы в реальном времени
- Показывает найденные релевантные документы
- Генерирует ответы с помощью LLM

**2. Демонстрационный режим**
```bash
python app/rag_system.py --mode demo --data output/knowledge-base.json
```
- Запускает предустановленные тестовые вопросы
- Показывает работу всех компонентов системы
- Выводит метрики производительности

#### Архитектура RAG системы

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF документ  │───▶│   pdf_to_qa.py   │───▶│  База знаний    │
│                 │    │                  │    │  (JSON файл)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Пользователь  │◀───│   rag_system.py  │◀───│  Векторное      │
│   (вопросы)     │    │                  │    │  хранилище      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   (генерация)    │
                       └──────────────────┘
```

#### Компоненты RAG системы

1. **Загрузчик базы знаний** - загружает и парсит JSON с Q&A парами
2. **TF-IDF векторное хранилище** - индексирует документы для ключевого поиска
3. **Семантический поиск** - использует эмбеддинги для семантического поиска
4. **Гибридный поиск** - комбинирует результаты TF-IDF и семантического поиска
5. **Ollama интеграция** - генерирует ответы с помощью локальной LLM
6. **Репозиторий векторного хранилища** - абстракция для легкой миграции между хранилищами

## 🔬 Исследовательские компоненты

В проекте также присутствуют скрипты для исследований в области обучения нейронных сетей. Эти компоненты можно не использовать для основной функциональности RAG системы:

- **fix_errors.py** - Исправление ошибок в данных (исследовательский)
- **merge_datasets.py** - Объединение датасетов (исследовательский)
- **train_qlora.py** - QLoRA обучение моделей (исследовательский)
- **continue_training.py** - Продолжение обучения (исследовательский)
- **test_trained_model.py** - Тестирование обученных моделей (исследовательский)
- **compare_models.py** - Сравнение моделей (исследовательский)

Эти скрипты представляют собой исследования методов машинного обучения для анализа промышленного оборудования и не являются частью основной RAG системы.

## 🚀 Примеры использования RAG системы

### Создание базы знаний из учебника
```bash
# 1. Создаем Q&A пары из PDF
python app/pdf_to_qa.py input/mining-textbook.pdf \
  --ollama-url https://193.247.73.14:11436 \
  --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF \
  -o output/mining_qa.json -q 20 --verbose

# 2. Запускаем RAG систему
python app/rag_system.py --mode interactive --data output/mining_qa.json
```

### Тестирование системы
```bash
# Полное тестирование
python app/test_rag.py --test all --data output/mining_qa.json

# Только тест загрузки базы знаний
python app/test_rag.py --test knowledge_base --data output/mining_qa.json
```

### Использование готовых баз знаний
```bash
# Список доступных баз знаний
python app/rag_system.py --list-data

# Использование готовой базы знаний
python app/rag_system.py --mode interactive --data output/base-mining-and-mining-quality.final.json
```

## 🚀 Полная установка и настройка

### 1. Подготовка системы
```bash
# Обновление системы
sudo apt update
sudo apt upgrade

# Установка Git
sudo apt install git

# Клонирование репозитория
git clone https://github.com/0xFFFA/superapp.git
cd superapp
```

### 2. Установка Python и виртуального окружения
```bash
# Установка пакета для виртуальных окружений (в зависимости от версии Python)
sudo apt install python3.12-venv  # Для Python 3.12
# или
sudo apt install python3.10-venv  # Для Python 3.10

# Создание и активация виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 3. Установка базовых моделей
```bash
# Создание каталога для моделей
mkdir -p models
cd models

# Установка git-lfs для скачивания больших файлов
sudo apt-get update
sudo apt-get install git-lfs
git lfs install

# Скачивание модели Qwen 2.5 3B
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct qwen-2.5-3b
cd ..
```

### 4. Установка Ollama
```bash
# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Скачивание модели для Ollama
ollama pull qwen2.5:3b
```

### 5. Установка драйверов NVIDIA (если есть GPU)
```bash
# Автоматическая установка рекомендуемых драйверов
sudo ubuntu-drivers install --gpgpu

# Перезагрузка системы
sudo reboot

# Проверка установки драйверов
nvidia-smi
```

### 6. Создание необходимых каталогов
```bash
mkdir -p output
mkdir -p input
```

## 📚 Процесс работы с RAG системой

### 1. Подготовка базы знаний из PDF
```bash
# Активация виртуального окружения
source venv/bin/activate

# Конвертация PDF в формат вопрос-ответ
python app/pdf_to_qa.py input/mining-textbook.pdf \
    -o output/mining_qa.json \
    -q 10 \
    --ollama-url https://193.247.73.14:11436 \
    --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF \
    --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest
```

### 2. Запуск RAG системы
```bash
# Интерактивный режим для работы с базой знаний
python app/rag_system.py --mode interactive --data output/mining_qa.json

# Демонстрационный режим для тестирования
python app/rag_system.py --mode demo --data output/mining_qa.json
```

### 3. Тестирование системы
```bash
# Полное тестирование всех компонентов
python app/test_rag.py --test all --data output/mining_qa.json

# Тест конкретного компонента
python app/test_rag.py --test knowledge_base --data output/mining_qa.json
```

### 4. Использование готовых баз знаний
```bash
# Просмотр доступных баз знаний
python app/rag_system.py --list-data

# Использование готовой базы знаний
python app/rag_system.py --mode interactive --data output/base-mining-and-mining-quality.final.json
```

### 5. Настройка Ollama сервера
```bash
# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest

# Запуск Ollama сервера
ollama serve
```

## 🎯 Основные возможности

### 🔍 RAG система
- **Гибридный поиск** - комбинация ключевого и семантического поиска
- **Интеллектуальные ответы** на основе найденной информации
- **Интерактивный режим** для работы в реальном времени
- **Демонстрационный режим** для тестирования

### 📚 Генерация базы знаний
- **Автоматическое извлечение** текста из PDF документов
- **Создание Q&A пар** с помощью LLM
- **Настраиваемые параметры** для качества генерации
- **Поддержка различных моделей** LLM

### 🔬 Исследовательские возможности
- **QLoRA обучение** с поддержкой последовательного обучения (исследовательский)
- **Первичное обучение** на новых доменах знаний (исследовательский)
- **Продолжение обучения** на дополнительных датасетах (исследовательский)
- **Адаптация** под различные технические области (исследовательский)

### 📊 Анализ и тестирование
- **Сравнение моделей** до и после обучения (исследовательский)
- **Оценка качества** ответов на технические вопросы (исследовательский)
- **Мониторинг** процесса обучения (исследовательский)
- **Валидация** результатов (исследовательский)

## 📋 Примеры использования RAG системы

### Создание базы знаний из PDF
```bash
# Базовое использование
python app/pdf_to_qa.py input/mining-textbook.pdf \
    -o output/mining_qa.json \
    -q 10 \
    --ollama-url https://193.247.73.14:11436 \
    --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF

# Расширенное использование с настройками
python app/pdf_to_qa.py input/technical_manual.pdf \
    -o output/manual_qa.json \
    -q 15 \
    --ollama-url https://193.247.73.14:11436 \
    --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF \
    --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest \
    --chunk-size 3000 \
    --verbose
```

### Запуск RAG системы
```bash
# Интерактивный режим
python app/rag_system.py --mode interactive --data output/mining_qa.json

# Демонстрационный режим
python app/rag_system.py --mode demo --data output/mining_qa.json

# С настройками модели
python app/rag_system.py \
    --mode interactive \
    --data output/mining_qa.json \
    --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest \
    --ollama-url https://193.247.73.14:11436 \
    --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF
```

### Тестирование системы
```bash
# Полное тестирование
python app/test_rag.py --test all --data output/mining_qa.json

# Тест конкретного компонента
python app/test_rag.py --test knowledge_base --data output/mining_qa.json

# Тест с настройками
python app/test_rag.py \
    --test all \
    --data output/mining_qa.json \
    --ollama-url https://193.247.73.14:11436 \
    --api-token k6Svw7EldnQLhBpivenz7E2Z01H8FF
```

### Использование готовых баз знаний
```bash
# Просмотр доступных баз знаний
python app/rag_system.py --list-data

# Использование готовой базы знаний
python app/rag_system.py \
    --mode interactive \
    --data output/base-mining-and-mining-quality.final.json

# Работа с несколькими базами знаний
python app/rag_system.py \
    --mode interactive \
    --data output/book-machine-equip.json
```

## 🔧 Установка и конфигурация RAG системы

### 📦 Поддерживаемые модели LLM

#### **Рекомендуемые модели для RAG**
- **YandexGPT 5 Lite 8B** - `yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest` (рекомендуется)
- **Qwen 2.5 3B** - `qwen2.5:3b` (легкая модель)
- **Mistral 7B** - `mistral:7b` (качественная модель)

#### **Модели для исследовательских скриптов**
- **Qwen 2.5 3B** - `Qwen/Qwen2.5-3B-Instruct` (~6GB)
- **Mistral 7B** - `mistralai/Mistral-7B-Instruct-v0.2` (~14GB)
- **YandexGPT 5 Lite 8B** - `yandex/YandexGPT-5-Lite-8B-instruct` (~16GB)

### 🚀 Установка Ollama и моделей

#### **Установка Ollama**
```bash
# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Запуск Ollama сервера
ollama serve
```

#### **Загрузка моделей для RAG**
```bash
# Рекомендуемая модель для RAG
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest

# Альтернативные модели
ollama pull qwen2.5:3b
ollama pull mistral:7b
```

#### **Проверка установки**
```bash
# Список установленных моделей
ollama list

# Проверка API
curl http://localhost:11434/api/tags
```

### 💾 Требования к системе

| Компонент | Минимальные требования | Рекомендуемые требования |
|-----------|----------------------|-------------------------|
| RAM | 8GB | 16GB+ |
| Диск | 20GB | 50GB+ |
| CPU | 4 ядра | 8+ ядер |
| GPU | Не требуется | 8GB+ VRAM |

### 🔧 Настройка для разных серверов

#### **Мощный сервер (16GB+ RAM)**
```bash
# Использовать YandexGPT 5 Lite 8B
python app/rag_system.py \
    --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest \
    --data output/knowledge-base.json
```

#### **Средний сервер (8GB RAM)**
```bash
# Использовать Qwen 2.5 3B
python app/rag_system.py \
    --model qwen2.5:3b \
    --data output/knowledge-base.json
```

#### **Слабый сервер (4GB RAM)**
```bash
# Использовать легкую модель
python app/rag_system.py \
    --model qwen2.5:3b \
    --data output/knowledge-base.json \
    --max-context 3
```

## ⚙️ Требования

### Для RAG системы
- **Python** 3.8+
- **Ollama** для работы с LLM
- **RAM** минимум 8GB (рекомендуется 16GB+)
- **Диск** минимум 20GB (рекомендуется 50GB+)
- **CPU** 4+ ядер

### Для исследовательских скриптов
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **PEFT** (Parameter-Efficient Fine-Tuning)
- **Git LFS** для скачивания моделей
- **GPU** с поддержкой CUDA (рекомендуется)
- **RAM** минимум 16GB
- **Диск** минимум 50GB для моделей

## 🔧 Конфигурация RAG системы

### Параметры поиска
- **Max Context**: 3-10 документов (по умолчанию 5)
- **Similarity Threshold**: 0.5-0.9 (по умолчанию 0.7)
- **Chunk Size**: 1000-4000 символов (по умолчанию 2000)
- **Overlap**: 100-500 символов (по умолчанию 200)

### Параметры генерации Q&A
- **Questions per Chunk**: 5-20 (по умолчанию 5)
- **Max Retries**: 1-5 (по умолчанию 3)
- **Timeout**: 10-60 секунд (по умолчанию 30)

### Рекомендуемые настройки по модели

#### **YandexGPT 5 Lite 8B (рекомендуется)**
```bash
python app/rag_system.py \
    --model yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest \
    --max-context 5 \
    --similarity-threshold 0.7
```

#### **Qwen 2.5 3B (легкая модель)**
```bash
python app/rag_system.py \
    --model qwen2.5:3b \
    --max-context 3 \
    --similarity-threshold 0.6
```

#### **Mistral 7B (качественная модель)**
```bash
python app/rag_system.py \
    --model mistral:7b \
    --max-context 7 \
    --similarity-threshold 0.8
```

## 📊 Мониторинг и логирование

### RAG система
- **Консольные логи** в реальном времени
- **Метрики поиска** - время поиска, количество найденных документов
- **Метрики генерации** - время генерации ответа, качество ответа
- **Статистика использования** - количество вопросов, популярные темы

### Исследовательские скрипты
- **Автосохранение** моделей каждые 100 шагов
- **Метрики качества** на каждом этапе обучения
- **Логи обучения** - loss, accuracy, learning rate

## 🔧 Исправления и улучшения

### ✅ Исправленные проблемы RAG системы

#### **SSL и аутентификация**
- ✅ Добавлена поддержка HTTPS для Ollama API
- ✅ Реализована аутентификация через X-Access-Token
- ✅ Отключены SSL предупреждения для самоподписанных сертификатов
- ✅ Добавлена поддержка различных моделей LLM

#### **Гибридный поиск**
- ✅ Реализован TF-IDF векторный поиск
- ✅ Добавлен семантический поиск с эмбеддингами
- ✅ Комбинированный гибридный поиск
- ✅ Настраиваемые пороги схожести

#### **Архитектура системы**
- ✅ Репозиторий векторного хранилища для легкой миграции
- ✅ Конфигурируемые параметры поиска и генерации
- ✅ Поддержка различных форматов баз знаний
- ✅ Интерактивный и демонстрационный режимы

### ✅ Исправленные проблемы исследовательских скриптов

#### **Ошибка "element 0 of tensors does not require_grad"**
- ✅ Добавлена поддержка Qwen моделей в `target_modules`
- ✅ Включение режима обучения (`model.train()`) после применения LoRA
- ✅ Отключение `use_cache` для совместимости с gradient checkpointing
- ✅ Проверка trainable параметров перед обучением

#### **Проблемы с продолжением обучения**
- ✅ Правильная загрузка LoRA адаптеров с `is_trainable=True`
- ✅ Принудительное включение градиентов для всех LoRA параметров
- ✅ Автоматическое исправление отсутствующих trainable параметров

#### **Оптимизация памяти**
- ✅ Настройка CUDA для лучшего управления памятью
- ✅ Ограничение потоков OpenMP, MKL, NumExpr
- ✅ Улучшенная токенизация с экономией памяти

## 🚨 Устранение неполадок

### Проблемы RAG системы

#### **SSL ошибки**
```bash
# Если используете самоподписанные сертификаты
export PYTHONHTTPSVERIFY=0

# Или добавьте verify=False в код
```

#### **Ошибки аутентификации Ollama**
```bash
# Проверьте токен доступа
curl -H "X-Access-Token: YOUR_TOKEN" https://your-server:11436/api/tags

# Проверьте URL сервера
curl https://your-server:11436/api/tags
```

#### **Проблемы с моделью**
```bash
# Проверьте доступные модели
ollama list

# Загрузите нужную модель
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest
```

#### **Медленный поиск**
- Уменьшите `--max-context` (по умолчанию 5)
- Увеличьте `--similarity-threshold` (по умолчанию 0.7)
- Используйте более мощный сервер

### Проблемы исследовательских скриптов

#### **Недостаточно памяти GPU**
```bash
# Уменьшить batch-size и lora-r
python app/train_qlora.py --batch-size 1 --lora-r 4

# Или использовать CPU
export CUDA_VISIBLE_DEVICES=""
```

#### **Недостаточно RAM**
```bash
# Уменьшить параметры
python app/train_qlora.py --epochs 1 --batch-size 1 --lora-r 4
```

#### **Ошибки скачивания моделей**
```bash
# Проверить git-lfs
git lfs install

# Проверить доступ к Hugging Face
curl -I https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
```

#### **Медленное обучение**
- Проверьте использование GPU: `nvidia-smi`
- Увеличьте batch-size если позволяет память
- Используйте более мощный сервер

### Поддержка
- **RAG система**: Проверьте логи в консоли, убедитесь в корректности путей к файлам
- **Исследовательские скрипты**: Проверьте совместимость версий библиотек
- **Все скрипты**: Имеют детальное логирование для диагностики проблем

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

Этот проект распространяется под лицензией MIT.

## 🙏 Благодарности

### RAG система
- **Ollama** за локальную работу с LLM
- **Yandex** за модель YandexGPT
- **Hugging Face** за библиотеки для работы с текстом
- **scikit-learn** за TF-IDF реализацию

### Исследовательские компоненты
- **Hugging Face** за библиотеки Transformers и PEFT
- **Microsoft** за LoRA метод
- **Alibaba Cloud** за модель Qwen
- **Сообщество open-source** за вклад в развитие

---

**SuperApp** - RAG система для горнодобывающей документации! 🚀
