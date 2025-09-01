#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для продолжения обучения уже обученной модели на новых данных
Поддерживает стратегию последовательного обучения на нескольких датасетах
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
    TaskType,
    PeftModel
)
from datasets import Dataset
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinueTrainingTrainer:
    """Тренер для продолжения обучения уже обученной модели"""
    
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-2-7b-hf",
                 trained_model_path: str = "trained_model",
                 output_dir: str = "trained_model_continued",
                 device: str = "auto"):
        """
        Инициализация тренера для продолжения обучения
        
        Args:
            base_model: Базовая модель (оригинальная)
            trained_model_path: Путь к уже обученной модели
            output_dir: Директория для сохранения обновленной модели
            device: Устройство для обучения (auto/cpu/cuda)
        """
        self.base_model = base_model
        self.trained_model_path = Path(trained_model_path)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Создаем выходную директорию
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем компоненты
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Инициализирован тренер для продолжения обучения")
        logger.info(f"Базовая модель: {base_model}")
        logger.info(f"Обученная модель: {trained_model_path}")
        logger.info(f"Выходная директория: {output_dir}")
    
    def load_trained_model_and_tokenizer(self):
        """Загружает уже обученную модель и токенизатор"""
        logger.info("Загружаю уже обученную модель...")
        
        try:
            # Загружаем токенизатор из обученной модели
            if (self.trained_model_path / "tokenizer.json").exists():
                logger.info("Загружаю токенизатор из обученной модели...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.trained_model_path,
                    trust_remote_code=True,
                    padding_side="right"
                )
            else:
                # Если токенизатора нет, загружаем из базовой модели
                logger.info("Загружаю токенизатор из базовой модели...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model,
                    trust_remote_code=True,
                    padding_side="right"
                )
            
            # Добавляем pad token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загружаем базовую модель
            logger.info("Загружаю базовую модель...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Подготавливаем модель для k-bit обучения
            base_model = prepare_model_for_kbit_training(base_model)
            
            # Загружаем LoRA адаптеры из обученной модели
            logger.info("Загружаю LoRA адаптеры...")
            if (self.trained_model_path / "adapter_config.json").exists():
                self.model = PeftModel.from_pretrained(base_model, self.trained_model_path)
                logger.info("LoRA адаптеры загружены успешно!")
            else:
                logger.warning("LoRA адаптеры не найдены, создаю новые...")
                self.setup_lora_config()
                self.model = get_peft_model(base_model, self.lora_config)
            
            # Проверяем, что модель готова к обучению
            self.model.print_trainable_parameters()
            
            logger.info("Модель загружена успешно!")
            
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
        
        # Определяем целевые модули автоматически
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
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
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлекаем пары вопрос-ответ
            if 'qa_pairs' in data:
                qa_pairs = data['qa_pairs']
            elif isinstance(data, list):
                qa_pairs = data
            else:
                raise ValueError("Неизвестная структура данных")
            
            # Формируем тексты для обучения
            training_texts = []
            for qa in qa_pairs:
                # Формируем промпт в формате инструкций
                text = f"### Инструкция: Проанализируй техническую документацию и ответь на вопрос.\n\n### Вопрос: {qa['question']}\n\n### Ответ: {qa['answer']}\n\n### Конец"
                training_texts.append(text)
            
            logger.info(f"Загружено {len(training_texts)} текстов для обучения")
            
            # Создаем Dataset
            dataset = Dataset.from_dict({"text": training_texts})
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise
    
    def tokenize_function(self, examples):
        """Токенизирует тексты для обучения"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def setup_training(self, 
                      train_dataset: Dataset,
                      learning_rate: float = 1e-4,  # Снижаем LR для продолжения обучения
                      num_epochs: int = 2,          # Меньше эпох для продолжения
                      batch_size: int = 4,
                      gradient_accumulation_steps: int = 4):
        """
        Настраивает параметры обучения
        
        Args:
            train_dataset: Датасет для обучения
            learning_rate: Скорость обучения (снижена для продолжения)
            num_epochs: Количество эпох (меньше для продолжения)
            batch_size: Размер батча
            gradient_accumulation_steps: Шаги накопления градиентов
        """
        logger.info("Настраиваю параметры обучения...")
        
        # Токенизируем датасет
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Настройки обучения (адаптированы для продолжения)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_strategy="steps",
            warmup_steps=50,  # Меньше warmup для продолжения
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            report_to=None,  # Отключаем wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Создаем тренер
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )
        
        logger.info("Обучение настроено!")
    
    def train(self):
        """Запускает обучение"""
        logger.info("Начинаю продолжение обучения...")
        
        try:
            # Запускаем обучение
            self.trainer.train()
            
            # Сохраняем модель
            self.save_model()
            
            logger.info("Продолжение обучения завершено успешно!")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise
    
    def save_model(self):
        """Сохраняет обновленную модель"""
        logger.info("Сохраняю обновленную модель...")
        
        try:
            # Сохраняем LoRA адаптеры
            self.model.save_pretrained(self.output_dir)
            
            # Сохраняем токенизатор
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Сохраняем конфигурацию с информацией о продолжении обучения
            config = {
                "base_model": self.base_model,
                "previous_training": str(self.trained_model_path),
                "training_info": {
                    "method": "QLoRA_Continued",
                    "output_dir": str(self.output_dir),
                    "type": "continued_training"
                }
            }
            
            with open(self.output_dir / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Обновленная модель сохранена в: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            raise


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Продолжение обучения уже обученной модели на новых данных"
    )
    parser.add_argument("--data", required=True, help="Путь к JSON файлу с новыми данными")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf", 
                       help="Базовая модель (оригинальная)")
    parser.add_argument("--trained-model", required=True,
                       help="Путь к уже обученной модели")
    parser.add_argument("--output-dir", default="trained_model_continued", 
                       help="Директория для сохранения обновленной модели")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Скорость обучения (снижена для продолжения)")
    parser.add_argument("--epochs", type=int, default=2, 
                       help="Количество эпох (меньше для продолжения)")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Размер батча")
    parser.add_argument("--lora-r", type=int, default=16, 
                       help="Ранг LoRA")
    
    args = parser.parse_args()
    
    # Проверяем существование файла с данными
    if not Path(args.data).exists():
        print(f"❌ Файл с данными не найден: {args.data}")
        return
    
    # Проверяем существование обученной модели
    if not Path(args.trained_model).exists():
        print(f"❌ Обученная модель не найдена: {args.trained_model}")
        return
    
    try:
        # Создаем тренер для продолжения обучения
        trainer = ContinueTrainingTrainer(
            base_model=args.base_model,
            trained_model_path=args.trained_model,
            output_dir=args.output_dir
        )
        
        # Загружаем уже обученную модель
        trainer.load_trained_model_and_tokenizer()
        
        # Настраиваем LoRA (если нужно)
        if not hasattr(trainer, 'lora_config'):
            trainer.setup_lora_config(r=args.lora_r)
        
        # Загружаем новые данные
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
        
        print(f"\n✅ Продолжение обучения завершено!")
        print(f"📁 Обновленная модель сохранена в: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
