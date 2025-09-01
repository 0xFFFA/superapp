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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.device = device
        
        # Создаем выходную директорию
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем компоненты
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        logger.info(f"Инициализирован тренер для модели: {base_model}")
    
    def load_model_and_tokenizer(self):
        """Загружает базовую модель и токенизатор"""
        logger.info("Загружаю модель и токенизатор...")
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Добавляем pad token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Подготавливаем модель для k-bit обучения
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info("Модель и токенизатор загружены успешно!")
            
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
        
        # Если target_modules не указаны, определяем автоматически по типу модели
        if target_modules is None:
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
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Применяем LoRA к модели
        self.model = get_peft_model(self.model, lora_config)
        
        # Печатаем информацию о trainable параметрах
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
        
        # Токенизируем датасет
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Настройки обучения
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
            warmup_steps=100,
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
    
    args = parser.parse_args()
    
    # Проверяем существование файла с данными
    if not Path(args.data).exists():
        print(f"❌ Файл не найден: {args.data}")
        return
    
    try:
        # Создаем тренер
        trainer = IndustrialQLoRATrainer(
            base_model=args.base_model,
            output_dir=args.output_dir
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
