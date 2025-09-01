#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для сравнения базовой и обученной моделей
Показывает разницу в ответах до и после обучения
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_models(base_model_path, lora_adapters_path):
    """Загружает базовую и обученную модели"""
    print("🔄 Загружаю модели...")
    
    # Базовая модель
    print(f"📥 Загружаю базовую модель: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Обученная модель с LoRA
    print(f"📥 Загружаю обученную модель: {lora_adapters_path}")
    trained_model = PeftModel.from_pretrained(base_model, lora_adapters_path)
    
    print("✅ Модели загружены!")
    return base_model, trained_model, tokenizer

def generate_answer(model, tokenizer, question, max_length=200, temperature=0.7):
    """Генерирует ответ на вопрос"""
    prompt = f"### Вопрос: {question}\n### Ответ:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только ответ
    if "### Ответ:" in answer:
        answer = answer.split("### Ответ:")[1].strip()
        if "###" in answer:
            answer = answer.split("###")[0].strip()
    
    return answer

def compare_models(base_model, trained_model, tokenizer, questions):
    """Сравнивает ответы базовой и обученной моделей"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("-" * 60)
        
        try:
            # Ответ базовой модели
            print("🔵 Базовая модель:")
            base_answer = generate_answer(base_model, tokenizer, question)
            print(f"   {base_answer}")
            
            print("\n🟢 Обученная модель:")
            trained_answer = generate_answer(trained_model, tokenizer, question)
            print(f"   {trained_answer}")
            
            # Анализ различий
            print("\n📊 Анализ:")
            if base_answer == trained_answer:
                print("   ⚠️  Ответы идентичны (возможно, модель не обучилась)")
            else:
                print("   ✅ Ответы различаются (модель обучилась)")
                if len(trained_answer) > len(base_answer):
                    print("   📈 Обученная модель дает более подробные ответы")
                elif len(trained_answer) < len(base_answer):
                    print("   📉 Обученная модель дает более краткие ответы")
            
        except Exception as e:
            print(f"❌ Ошибка при генерации: {e}")
        
        print("\n" + "="*60)

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Сравнение базовой и обученной моделей"
    )
    parser.add_argument(
        "--base-model", 
        default="/home/dev/industrial-ai-trainer/models/qwen-2.5-3b",
        help="Путь к базовой модели (по умолчанию: qwen-2.5-3b)"
    )
    parser.add_argument(
        "--trained-model", 
        default="./trained_model/v1",
        help="Путь к обученной модели (по умолчанию: ./trained_model/v1)"
    )
    parser.add_argument(
        "--questions", 
        nargs="+",
        default=[
            "Что представляет собой шаровая мельница?",
            "Какой тип диспергаторов получил наибольшее распространение?",
            "Какое значение имеет диаметр шаров при диспергировании?",
            "Какие преимущества имеет шаровая мельница?",
            "Как работает шаровая мельница?"
        ],
        help="Вопросы для сравнения (по умолчанию: стандартные вопросы)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=200,
        help="Максимальная длина ответа (по умолчанию: 200)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Температура генерации (по умолчанию: 0.7)"
    )
    
    args = parser.parse_args()
    
    print("🏭 Сравнение базовой и обученной моделей")
    print(f"🔵 Базовая модель: {args.base_model}")
    print(f"🟢 Обученная модель: {args.trained_model}")
    print(f"❓ Количество вопросов: {len(args.questions)}")
    print(f"🌡️  Температура: {args.temperature}")
    print(f"📏 Максимальная длина: {args.max_length}")
    print("-" * 80)
    
    try:
        # Загружаем модели
        base_model, trained_model, tokenizer = load_models(args.base_model, args.trained_model)
        
        # Сравниваем модели
        compare_models(base_model, trained_model, tokenizer, args.questions)
        
        print("\n" + "="*80)
        print("СРАВНЕНИЕ ЗАВЕРШЕНО")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
