#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования обученной модели
Тестирует модель на вопросах из датасета и новых вопросах
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path

def load_trained_model(base_model_path, lora_adapters_path):
    """
    Загружает обученную модель с LoRA адаптерами
    
    Args:
        base_model_path: Путь к базовой модели
        lora_adapters_path: Путь к LoRA адаптерам
    
    Returns:
        model, tokenizer: Загруженная модель и токенизатор
    """
    print(f"Загружаю базовую модель из: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Загружаю LoRA адаптеры из: {lora_adapters_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapters_path)
    
    return model, tokenizer

def test_model(model, tokenizer, questions, max_length=200, temperature=0.7):
    """
    Тестирует модель на списке вопросов
    
    Args:
        model: Обученная модель
        tokenizer: Токенизатор
        questions: Список вопросов для тестирования
        max_length: Максимальная длина ответа
        temperature: Температура генерации
    """
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("-" * 40)
        
        try:
            # Формируем промпт
            prompt = f"### Вопрос: {question}\n### Ответ:"
            
            # Токенизируем
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ (убираем промпт и все после следующего ###)
            if "### Ответ:" in answer:
                answer = answer.split("### Ответ:")[1].strip()
                # Убираем все после следующего ### если он есть
                if "###" in answer:
                    answer = answer.split("###")[0].strip()
            
            print(f"Ответ: {answer}")
            
        except Exception as e:
            print(f"Ошибка при генерации: {e}")

def main():
    """Основная функция"""
    
    # Пути к моделям
    base_model_path = "/home/dev/industrial-ai-trainer/models/qwen-2.5-3b"
    lora_adapters_path = "./trained_model_mistral"  # Относительный путь из каталога training
    
    # Проверяем существование путей
    if not Path(base_model_path).exists():
        print(f"❌ Базовая модель не найдена: {base_model_path}")
        return
    
    if not Path(lora_adapters_path).exists():
        print(f"❌ LoRA адаптеры не найдены: {lora_adapters_path}")
        print("Убедитесь, что обучение завершено и модель сохранена")
        return
    
    try:
        # Загружаем модель
        print("🚀 Загружаю обученную модель...")
        model, tokenizer = load_trained_model(base_model_path, lora_adapters_path)
        print("✅ Модель загружена успешно!")
        
        # Вопросы для тестирования
        test_questions = [
            # Вопросы из исходного датасета
            "Что представляет собой шаровая мельница?",
            "Какой тип диспергаторов получил наибольшее распространение?",
            "Какое значение имеет диаметр шаров при диспергировании?",
            "Как можно рассчитать частоту вращения барабана для лавинообразного режима движения шаров?",
            
            # Новые вопросы для проверки обобщения
            "Какие преимущества имеет шаровая мельница по сравнению с другими диспергаторами?",
            "Как оптимизировать производительность шаровой мельницы?",
            "В каких случаях лучше использовать шаровую мельницу?",
            "Какие факторы влияют на эффективность работы шаровой мельницы?"
        ]
        
        # Тестируем модель
        test_model(model, tokenizer, test_questions)
        
        print("\n" + "="*60)
        print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
