#!/bin/bash

# Скрипт для запуска обучения с лимитами памяти

# Устанавливаем лимиты для оперативной памяти
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Ограничиваем количество процессов
ulimit -u 1000  # Лимит на количество процессов

# Запускаем обучение с ограниченными ресурсами
echo "🚀 Запуск обучения с лимитами памяти..."
echo "📊 Лимиты:"
echo "   - OpenMP threads: $OMP_NUM_THREADS"
echo "   - MKL threads: $MKL_NUM_THREADS"
echo "   - NumExpr threads: $NUMEXPR_NUM_THREADS"
echo "   - Процессы: $(ulimit -u)"
echo ""

# Активируем виртуальное окружение и запускаем обучение
source venv/bin/activate

# Запускаем обучение с консервативными параметрами
python app/train_qlora.py \
  --data output/ballmill-part3.final.json \
  --base-model models/qwen-2.5-3b \
  --output-dir trained_model/v1 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --lora-r 4 \
  --device cuda

echo "✅ Обучение завершено!"
