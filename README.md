# 🏭 SuperApp - Industrial AI Trainer

Система для обучения и анализа промышленных языковых моделей на основе технической документации.

## 🎯 Назначение

Этот проект предназначен для создания специализированных AI-моделей, способных анализировать техническую документацию промышленного оборудования, отвечать на вопросы о технологических процессах и помогать инженерам в решении технических задач.

## ✨ Ключевые возможности

- 🔄 **Конвертация PDF в Q&A** - автоматическое создание датасетов из технической документации
- 🎓 **QLoRA обучение** - эффективное fine-tuning с минимальными ресурсами
- 🔄 **Продолжение обучения** - последовательное улучшение моделей на новых данных
- 📊 **Сравнение моделей** - оценка качества обучения
- 🛠️ **Исправления ошибок** - автоматическая обработка проблемных данных
- 📈 **Мониторинг** - детальное логирование и отслеживание прогресса

## 🚀 Быстрый старт

1. **Установка**: `sudo apt update && sudo apt install git && git clone https://github.com/0xFFFA/superapp.git`
2. **Настройка**: `cd superapp && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. **Модели**: `mkdir -p models && cd models && git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct qwen-2.5-3b`
4. **Ollama**: `curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:3b`
5. **Обучение**: `python app/train_qlora.py --data output/dataset.json --base-model models/qwen-2.5-3b --output-dir trained_model/v1`

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

## 📚 Процесс обучения модели

### 1. Подготовка датасета из PDF
```bash
# Активация виртуального окружения
source venv/bin/activate

# Конвертация PDF в формат вопрос-ответ
python app/pdf_to_qa.py input/ballmill-part3.pdf \
    -o output/ballmill-part3.json \
    -m qwen2.5:3b \
    -q 7 \
    --ollama-url http://localhost:11434
```

### 2. Обработка ошибок (если необходимо)
```bash
# Если в каталоге output появился файл ballmill-part3.err
python app/fix_errors.py output/ballmill-part3.err \
    -o output/ballmill-part3.mid
```

### 3. Объединение датасетов
```bash
# Объединение файлов json и mid в один файл
python app/merge_datasets.py \
    output/ballmill-part3.json \
    output/ballmill-part3.mid \
    -o output/ballmill-part3.final.json
```

### 4. Обучение модели
```bash
# Для систем без GPU или со слабым GPU
export CUDA_VISIBLE_DEVICES=""

# Обучение с оптимальными параметрами
python app/train_qlora.py \
    --data output/ballmill-part3.final.json \
    --base-model models/qwen-2.5-3b \
    --output-dir trained_model/v1 \
    --learning-rate 1e-4 \
    --epochs 5 \
    --batch-size 2 \
    --lora-r 32 \
    --device cuda
```

### 5. Сравнение моделей
```bash
# Сравнение базовой и обученной моделей
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --questions \
        "Что такое шаровая мельница?" \
        "Как работает диспергатор?" \
        "Какие факторы влияют на эффективность измельчения?"
```

### 6. Продолжение обучения
```bash
# Продолжение обучения на новых данных
python app/continue_training.py \
    --data output/new_dataset.json \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --output-dir trained_model/v2 \
    --learning-rate 5e-5 \
    --epochs 2 \
    --batch-size 2 \
    --lora-r 16 \
    --device cuda
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
# Базовое использование
python app/pdf_to_qa.py input/document.pdf \
    -o output/dataset.json \
    -m qwen2.5:3b \
    -q 5

# Расширенное использование с настройками
python app/pdf_to_qa.py input/technical_manual.pdf \
    -o output/manual_qa.json \
    -m qwen2.5:3b \
    -q 10 \
    --ollama-url http://localhost:11434 \
    --chunk-size 3000
```

### Обучение модели
```bash
# Быстрое обучение для тестирования
python app/train_qlora.py \
    --data output/dataset.json \
    --base-model models/qwen-2.5-3b \
    --output-dir trained_model/v1_quick \
    --epochs 1 \
    --learning-rate 2e-4 \
    --batch-size 4 \
    --lora-r 8

# Качественное обучение
python app/train_qlora.py \
    --data output/dataset.json \
    --base-model models/qwen-2.5-3b \
    --output-dir trained_model/v1_quality \
    --epochs 5 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --lora-r 32
```

### Продолжение обучения
```bash
# Продолжение на новых данных
python app/continue_training.py \
    --data output/new_data.json \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --output-dir trained_model/v2 \
    --learning-rate 5e-5 \
    --epochs 2

# Адаптация под новый домен
python app/continue_training.py \
    --data output/chemical_data.json \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --output-dir trained_model/chemical \
    --learning-rate 1e-4 \
    --epochs 3
```

### Сравнение моделей
```bash
# Базовое сравнение
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1

# Сравнение с кастомными вопросами
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --questions \
        "Как работает насос?" \
        "Что такое диспергатор?" \
        "Какие факторы влияют на эффективность?"

# Сравнение с настройками генерации
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v1 \
    --questions \
        "Объясните принцип работы шаровой мельницы" \
        "Какие материалы используются для изготовления крепи?" \
    --max-length 300 \
    --temperature 0.7
```

### Тестирование на горном деле
```bash
# Тестирование с подготовленными вопросами по горному делу
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained_model/v3_continued \
    --questions \
        "Какие типы горных выработок используются в подземной добыче?" \
        "Как определяется коэффициент крепости Протодьяконова?" \
        "Какие требования предъявляются к горной крепи?" \
        "Как рассчитывается удельный расход взрывчатых веществ?" \
        "Какие методы используются для проветривания выработок?" \
    --max-length 300
```

## 🔧 Установка и конфигурация моделей

### 📦 Поддерживаемые модели

#### **Публичные модели (без авторизации)**
- **Qwen 2.5 3B** - `Qwen/Qwen2.5-3B-Instruct` (~6GB)
- **Mistral 7B** - `mistralai/Mistral-7B-Instruct-v0.2` (~14GB)
- **YandexGPT 5 Lite 8B** - `yandex/YandexGPT-5-Lite-8B-instruct` (~16GB)

#### **Модели с авторизацией**
- **Llama 2** - `meta-llama/Llama-2-7b-chat-hf` (требует токен)
- **CodeLlama** - `codellama/CodeLlama-7b-Instruct-hf`

### 🚀 Быстрая установка моделей

#### **Автоматическая установка**
```bash
# Скрипт для установки всех моделей
./install_models.sh
```

#### **Ручная установка**
```bash
# 1. Установить git-lfs
sudo apt-get install git-lfs
git lfs install

# 2. Скачать модели
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/qwen-2.5-3b
git clone https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct models/yandex-model

# 3. Проверить установку
ls -la models/
```

### 💾 Требования к дисковому пространству

| Модель | Размер | Рекомендуемый GPU |
|--------|--------|-------------------|
| Qwen 2.5 3B | ~6GB | 8GB+ |
| YandexGPT 5 Lite 8B | ~16GB | 16GB+ |
| Mistral 7B | ~14GB | 16GB+ |
| Llama 2 7B | ~13GB | 16GB+ |

### 🔧 Настройка для разных серверов

#### **Мощный сервер (GPU 16GB+)**
```bash
# Можно использовать любые модели
python app/train_qlora.py \
    --base-model models/yandex-model \
    --batch-size 4 \
    --lora-r 16
```

#### **Средний сервер (GPU 8GB)**
```bash
# Использовать Qwen 2.5 3B
python app/train_qlora.py \
    --base-model models/qwen-2.5-3b \
    --batch-size 2 \
    --lora-r 8
```

#### **Слабый сервер (CPU только)**
```bash
# Отключить GPU
export CUDA_VISIBLE_DEVICES=""

python app/train_qlora.py \
    --base-model models/qwen-2.5-3b \
    --batch-size 1 \
    --lora-r 4 \
    --epochs 1
```

## ⚙️ Требования

- **Python** 3.8+
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **PEFT** (Parameter-Efficient Fine-Tuning)
- **Git LFS** для скачивания моделей
- **GPU** с поддержкой CUDA (рекомендуется)
- **RAM** минимум 16GB
- **Диск** минимум 50GB для моделей

## 🔧 Конфигурация

### Параметры обучения
- **LoRA Rank**: 4-64 (по умолчанию 16)
- **Learning Rate**: 1e-5 до 2e-4
- **Batch Size**: 1-16 (зависит от GPU)
- **Epochs**: 1-10 (зависит от задачи)

### Рекомендуемые настройки по модели

#### **Qwen 2.5 3B**
```bash
--epochs 3 --learning-rate 2e-4 --batch-size 2 --lora-r 16
```

#### **YandexGPT 5 Lite 8B**
```bash
--epochs 2 --learning-rate 1e-4 --batch-size 1 --lora-r 8
```

#### **Mistral 7B**
```bash
--epochs 2 --learning-rate 1e-4 --batch-size 1 --lora-r 8
```

## 📊 Мониторинг и логирование

- **Консольные логи** в реальном времени
- **Автосохранение** моделей каждые 100 шагов
- **Метрики качества** на каждом этапе

## 🔧 Исправления и улучшения

### ✅ Исправленные проблемы

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

### Частые проблемы

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

#### **Проблемы с Ollama**
```bash
# Проверить статус Ollama
ollama list

# Перезапустить Ollama
sudo systemctl restart ollama

# Проверить доступность API
curl http://localhost:11434/api/tags
```

### Поддержка
- Проверьте логи в консоли
- Убедитесь в корректности путей к файлам
- Проверьте совместимость версий библиотек
- Все скрипты теперь имеют детальное логирование

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
- Alibaba Cloud за модель Qwen
- Yandex за модель YandexGPT
- Сообщество open-source за вклад в развитие

---

**SuperApp** - Создаем будущее промышленной аналитики с помощью AI! 🚀
