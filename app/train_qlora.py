#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA обучение для промышленной нейросети
Обучает модель на технической документации для анализа промышленного оборудования
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import logging
import psutil  # Для мониторинга памяти

# Настройка CUDA для лучшего управления памятью
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Лимиты для оперативной памяти
os.environ['OMP_NUM_THREADS'] = '8'  # Ограничиваем OpenMP потоки
os.environ['MKL_NUM_THREADS'] = '8'  # Ограничиваем MKL потоки
os.environ['NUMEXPR_NUM_THREADS'] = '8'  # Ограничиваем NumExpr потоки

# Ограничение использования RAM для PyTorch
import torch
torch.set_num_threads(8)  # Ограничиваем количество потоков PyTorch

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)
logger = logging.getLogger(__name__)


def log_memory_usage():
    """Логирует использование памяти"""
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {memory.percent:.1f}% используется ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_memory:.1f}GB / {gpu_total:.1f}GB")


class IndustrialQLoRATrainer:
    """QLoRA тренер для промышленной нейросети"""
    
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-2-7b-hf",
                 output_dir: str = "trained_model",
                 device: str = "auto"):
        """
        Инициализация тренера
        
        Args:
            base_model: Базовая модель для обучения
            output_dir: Директория для сохранения обученной модели
            device: Устройство для обучения (auto/cpu/cuda)
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        
        # Определяем устройство автоматически
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Используется GPU (CUDA)")
            else:
                self.device = "cpu"
                logger.info("Используется CPU")
        else:
            self.device = device
            logger.info(f"Используется устройство: {device}")
        
        # Создаем выходную директорию
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем компоненты
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Инициализирован тренер для модели: {base_model}")
        log_memory_usage()  # Логируем использование памяти
    
    def load_model_and_tokenizer(self):
        """Загружает базовую модель и токенизатор"""
        logger.info("Загружаю модель и токенизатор...")
        
        try:
            # Загружаем токенизатор
            logger.info("Загружаю токенизатор...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            logger.info("Токенизатор загружен успешно")
            
            # Добавляем pad token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Установлен pad_token = eos_token")
            
            # Загружаем модель
            logger.info("Загружаю модель...")
            
            # Настройки загрузки в зависимости от устройства
            if self.device == "cpu":
                # Для CPU используем float32 и не используем device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                # Перемещаем модель на CPU
                self.model = self.model.to("cpu")
                logger.info("Модель загружена успешно")
                # Для CPU не используем k-bit подготовку
                logger.info("Модель подготовлена для обучения на CPU")
            else:
                # Для GPU используем float16 и device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                    max_memory={0: "14GB"},  # Ограничиваем использование памяти GPU
                    low_cpu_mem_usage=True  # Экономия памяти CPU
                )
                logger.info("Модель загружена успешно")
                
                # Подготавливаем модель для k-bit обучения
                logger.info("Подготавливаю модель для k-bit обучения...")
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Модель подготовлена для k-bit обучения")
            
            logger.info("Модель и токенизатор загружены успешно!")
            log_memory_usage()  # Логируем использование памяти после загрузки модели
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def setup_lora_config(self, 
                          r: int = 16,
                          lora_alpha: int = 32,
                          lora_dropout: float = 0.1,
                          target_modules: Optional[List[str]] = None):
        """
        Настраивает LoRA конфигурацию
        
        Args:
            r: Ранг LoRA
            lora_alpha: Альфа параметр LoRA
            lora_dropout: Dropout для LoRA
            target_modules: Целевые модули для LoRA
        """
        logger.info("Настраиваю LoRA конфигурацию...")
        logger.info(f"Параметры LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # Если target_modules не указаны, определяем автоматически по типу модели
        if target_modules is None:
            logger.info("Определяю target_modules автоматически...")
            if "llama" in self.base_model.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif "gpt" in self.base_model.lower():
                target_modules = ["c_attn", "c_proj", "c_fc", "c_proj"]
            elif "bert" in self.base_model.lower():
                target_modules = ["query", "key", "value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            else:
                # Универсальные модули для большинства трансформеров
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        logger.info(f"Используемые target_modules: {target_modules}")
        
        logger.info("Создаю LoraConfig...")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        logger.info("LoraConfig создан успешно")
        
        # Применяем LoRA к модели
        logger.info("Применяю LoRA к модели...")
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA применена к модели")
        
        # Печатаем информацию о trainable параметрах
        logger.info("Информация о trainable параметрах:")
        self.model.print_trainable_parameters()
        
        logger.info("LoRA конфигурация настроена!")
    
    def load_training_data(self, data_path: str) -> Dataset:
        """
        Загружает данные для обучения
        
        Args:
            data_path: Путь к JSON файлу с данными
            
        Returns:
            Dataset для обучения
        """
        logger.info(f"Загружаю данные из: {data_path}")
        
        try:
            logger.info("Открываю JSON файл...")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info("JSON файл успешно загружен")
            
            # Извлекаем пары вопрос-ответ
            logger.info("Анализирую структуру данных...")
            if 'qa_pairs' in data:
                qa_pairs = data['qa_pairs']
                logger.info(f"Найдены qa_pairs: {len(qa_pairs)} пар")
            elif isinstance(data, list):
                qa_pairs = data
                logger.info(f"Данные в формате списка: {len(qa_pairs)} элементов")
            else:
                logger.error(f"Неизвестная структура данных. Ключи: {list(data.keys()) if isinstance(data, dict) else 'не словарь'}")
                raise ValueError("Неизвестная структура данных")
            
            # Формируем тексты для обучения
            logger.info("Формирую тексты для обучения...")
            training_texts = []
            for i, qa in enumerate(qa_pairs):
                # Формируем промпт в формате инструкций
                text = f"### Инструкция: Проанализируй техническую документацию и ответь на вопрос.\n\n### Вопрос: {qa['question']}\n\n### Ответ: {qa['answer']}\n\n### Конец"
                training_texts.append(text)
                if i % 100 == 0 and i > 0:
                    logger.info(f"Обработано {i}/{len(qa_pairs)} пар")
            
            logger.info(f"Загружено {len(training_texts)} текстов для обучения")
            
            # Создаем Dataset
            logger.info("Создаю Dataset...")
            dataset = Dataset.from_dict({"text": training_texts})
            logger.info(f"Dataset создан успешно. Размер: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise
    
    def tokenize_function(self, examples):
        """Токенизирует тексты для обучения"""
        logger.debug(f"Токенизирую батч из {len(examples['text'])} примеров")
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None  # Не возвращаем тензоры PyTorch для экономии памяти
        )
        logger.debug(f"Токенизация завершена. Размеры: {len(result['input_ids'])} примеров")
        return result
    
    def setup_training(self, 
                      train_dataset: Dataset,
                      learning_rate: float = 2e-4,
                      num_epochs: int = 3,
                      batch_size: int = 4,
                      gradient_accumulation_steps: int = 4):
        """
        Настраивает параметры обучения
        
        Args:
            train_dataset: Датасет для обучения
            learning_rate: Скорость обучения
            num_epochs: Количество эпох
            batch_size: Размер батча
            gradient_accumulation_steps: Шаги накопления градиентов
        """
        logger.info("Настраиваю параметры обучения...")
        logger.info(f"Размер датасета: {len(train_dataset)} примеров")
        logger.info(f"Параметры обучения: lr={learning_rate}, epochs={num_epochs}, batch_size={batch_size}, grad_accum={gradient_accumulation_steps}")
        
        # Токенизируем датасет
        logger.info("Начинаю токенизацию датасета...")
        try:
            # Очищаем память перед токенизацией
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Используем меньший batch_size для токенизации и отключаем return_tensors
            tokenized_dataset = train_dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=10,  # Уменьшаем размер батча для токенизации
                remove_columns=train_dataset.column_names
            )
            logger.info(f"Токенизация завершена. Размер токенизированного датасета: {len(tokenized_dataset)}")
        except Exception as e:
            logger.error(f"Ошибка при токенизации: {e}")
            raise
        
        # Настройки обучения
        logger.info("Создаю TrainingArguments...")
        try:
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                fp16=(self.device != "cpu"),  # Отключаем fp16 для CPU
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                save_strategy="steps",
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=str(self.output_dir / "logs"),
                report_to=None,  # Отключаем wandb/tensorboard
                remove_unused_columns=False,
                dataloader_pin_memory=(self.device != "cpu"),  # Отключаем pin_memory для CPU
                dataloader_num_workers=0,  # Отключаем многопроцессорность для экономии памяти
                gradient_checkpointing=(self.device != "cpu"),  # Отключаем gradient checkpointing для CPU
            )
            logger.info("TrainingArguments созданы успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании TrainingArguments: {e}")
            raise
        
        # Создаем тренер
        logger.info("Создаю Trainer...")
        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                ),
            )
            logger.info("Trainer создан успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании Trainer: {e}")
            raise
        
        logger.info("Обучение настроено!")
    
    def train(self):
        """Запускает обучение"""
        logger.info("Начинаю обучение...")
        
        try:
            # Запускаем обучение
            self.trainer.train()
            
            # Сохраняем модель
            self.save_model()
            
            logger.info("Обучение завершено успешно!")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise
    
    def save_model(self):
        """Сохраняет обученную модель"""
        logger.info("Сохраняю модель...")
        
        try:
            # Сохраняем LoRA адаптеры
            self.model.save_pretrained(self.output_dir)
            
            # Сохраняем токенизатор
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Сохраняем конфигурацию
            config = {
                "base_model": self.base_model,
                "training_info": {
                    "method": "QLoRA",
                    "output_dir": str(self.output_dir)
                }
            }
            
            with open(self.output_dir / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Модель сохранена в: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            raise


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="QLoRA обучение для промышленной нейросети"
    )
    parser.add_argument("--data", required=True, help="Путь к JSON файлу с данными")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf", 
                       help="Базовая модель для обучения")
    parser.add_argument("--output-dir", default="trained_model", 
                       help="Директория для сохранения")
    parser.add_argument("--learning-rate", type=float, default=2e-4, 
                       help="Скорость обучения")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Размер батча")
    parser.add_argument("--lora-r", type=int, default=16, 
                       help="Ранг LoRA")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cpu", "cuda"],
                       help="Устройство для обучения (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Проверяем существование файла с данными
    if not Path(args.data).exists():
        print(f"❌ Файл не найден: {args.data}")
        return
    
    try:
        # Создаем тренер
        trainer = IndustrialQLoRATrainer(
            base_model=args.base_model,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # Загружаем модель и токенизатор
        trainer.load_model_and_tokenizer()
        
        # Настраиваем LoRA
        trainer.setup_lora_config(r=args.lora_r)
        
        # Загружаем данные
        train_dataset = trainer.load_training_data(args.data)
        
        # Настраиваем обучение
        trainer.setup_training(
            train_dataset=train_dataset,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Запускаем обучение
        trainer.train()
        
        print(f"✅ Обучение завершено! Модель сохранена в: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return


if __name__ == "__main__":
    main()
