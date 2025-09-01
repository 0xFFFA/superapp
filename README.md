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
git clone https://github.com/0xFFFA/superapp.git
cd superapp
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Установка базовых моделей
```bash
# Создать каталог для моделей
mkdir -p models

# Установить git-lfs для скачивания больших файлов
sudo apt-get install git-lfs  # Ubuntu/Debian
# или
brew install git-lfs          # macOS

git lfs install
```

### 3. Скачивание моделей

#### **Qwen 2.5 3B (рекомендуется для начала)**
```bash
# Скачать Qwen модель (~6GB)
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/qwen-2.5-3b
```

#### **YandexGPT 5 Lite 8B (для русского языка)**
```bash
# Скачать Yandex модель (~16GB)
git clone https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct models/yandex-model
```

#### **Другие модели**
```bash
# Mistral 7B
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 models/mistral-7b

# Llama 2 7B (требует авторизацию)
huggingface-cli login
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf models/llama2-7b
```

### 4. Подготовка данных
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

### 5. Обучение модели
```bash
# Обучение на Qwen модели
python app/train_qlora.py \
    --data output/dataset.json \
    --base-model models/qwen-2.5-3b \
    --output-dir trained-models/model_v1 \
    --epochs 3 \
    --learning-rate 2e-4

# Обучение на Yandex модели (требует больше памяти)
python app/train_qlora.py \
    --data output/dataset.json \
    --base-model models/yandex-model \
    --output-dir trained-models/yandex_v1 \
    --epochs 2 \
    --learning-rate 2e-4 \
    --batch-size 1
```

### 6. Тестирование модели
```bash
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained-models/model_v1
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
    --base-model models/qwen-2.5-3b \
    --output-dir trained-models/model_v1 \
    --epochs 3 \
    --learning-rate 2e-4
```

### Продолжение обучения
```bash
python app/continue_training.py \
    --data input/new_data.json \
    --base-model models/qwen-2.5-3b \
    --trained-model trained-models/model_v1 \
    --output-dir trained-models/model_v2
```

### Сравнение моделей
```bash
python app/compare_models.py \
    --base-model models/qwen-2.5-3b \
    --trained-model trained-models/model_v1 \
    --questions "Как работает насос?" "Что такое диспергатор?"
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
- Alibaba Cloud за модель Qwen
- Yandex за модель YandexGPT
- Сообщество open-source за вклад в развитие

---

**SuperApp** - Создаем будущее промышленной аналитики с помощью AI! 🚀
