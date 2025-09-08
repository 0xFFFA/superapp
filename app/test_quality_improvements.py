#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование улучшений качества обучения
Скрипт для проверки эффективности улучшений
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Импортируем наши улучшения
from prompt_templates import PromptTemplates, validate_qa_quality, create_enhanced_training_text


class QualityTester:
    """Класс для тестирования улучшений качества"""
    
    def __init__(self):
        self.results = {}
    
    def test_prompt_templates(self, test_data: List[Dict]) -> Dict:
        """
        Тестирует различные промпт-шаблоны
        
        Args:
            test_data: Тестовые данные вопрос-ответ
            
        Returns:
            Результаты тестирования
        """
        print("🧪 Тестирую промпт-шаблоны...")
        
        results = {
            "enhanced": [],
            "context_aware": [],
            "multi_step": []
        }
        
        for qa in test_data[:5]:  # Тестируем на первых 5 примерах
            question = qa['question']
            answer = qa['answer']
            
            # Тестируем разные шаблоны
            for template_type in results.keys():
                try:
                    enhanced_text = create_enhanced_training_text(question, answer, template_type)
                    results[template_type].append({
                        "question": question,
                        "enhanced_text": enhanced_text,
                        "length": len(enhanced_text)
                    })
                except Exception as e:
                    print(f"Ошибка с шаблоном {template_type}: {e}")
        
        return results
    
    def test_data_validation(self, test_data: List[Dict]) -> Dict:
        """
        Тестирует валидацию качества данных
        
        Args:
            test_data: Тестовые данные
            
        Returns:
            Результаты валидации
        """
        print("🔍 Тестирую валидацию данных...")
        
        validation_results = {
            "total": len(test_data),
            "passed": 0,
            "failed": 0,
            "issues": [],
            "scores": []
        }
        
        for i, qa in enumerate(test_data):
            validation = validate_qa_quality(qa['question'], qa['answer'])
            validation_results['scores'].append(validation['quality_score'])
            
            if validation['passed']:
                validation_results['passed'] += 1
            else:
                validation_results['failed'] += 1
                validation_results['issues'].extend(validation['issues'])
        
        # Вычисляем статистику
        if validation_results['scores']:
            validation_results['avg_score'] = sum(validation_results['scores']) / len(validation_results['scores'])
            validation_results['min_score'] = min(validation_results['scores'])
            validation_results['max_score'] = max(validation_results['scores'])
        
        return validation_results
    
    def compare_old_vs_new_prompts(self, test_data: List[Dict]) -> Dict:
        """
        Сравнивает старые и новые промпты
        
        Args:
            test_data: Тестовые данные
            
        Returns:
            Результаты сравнения
        """
        print("📊 Сравниваю старые и новые промпты...")
        
        comparison = {
            "old_prompts": [],
            "new_prompts": [],
            "improvements": []
        }
        
        for qa in test_data[:3]:  # Тестируем на первых 3 примерах
            question = qa['question']
            answer = qa['answer']
            
            # Старый промпт
            old_prompt = f"### Инструкция: Проанализируй техническую документацию и ответь на вопрос.\n\n### Вопрос: {question}\n\n### Ответ: {answer}\n\n### Конец"
            
            # Новый промпт
            new_prompt = create_enhanced_training_text(question, answer, "enhanced")
            
            comparison['old_prompts'].append({
                "question": question,
                "prompt": old_prompt,
                "length": len(old_prompt)
            })
            
            comparison['new_prompts'].append({
                "question": question,
                "prompt": new_prompt,
                "length": len(new_prompt)
            })
            
            # Анализируем улучшения
            improvement = {
                "question": question,
                "length_increase": len(new_prompt) - len(old_prompt),
                "has_context": "эксперт" in new_prompt.lower(),
                "has_specialization": "специализация" in new_prompt.lower(),
                "has_technical_terms": any(term in new_prompt.lower() for term in ["техническая", "документация", "промышленное"])
            }
            comparison['improvements'].append(improvement)
        
        return comparison
    
    def generate_quality_report(self, test_data: List[Dict]) -> str:
        """
        Генерирует отчет о качестве
        
        Args:
            test_data: Тестовые данные
            
        Returns:
            Отчет в виде строки
        """
        print("📋 Генерирую отчет о качестве...")
        
        # Тестируем все компоненты
        prompt_results = self.test_prompt_templates(test_data)
        validation_results = self.test_data_validation(test_data)
        comparison_results = self.compare_old_vs_new_prompts(test_data)
        
        # Формируем отчет
        report = f"""
# 📊 ОТЧЕТ О КАЧЕСТВЕ УЛУЧШЕНИЙ

## 🎯 Общая статистика
- Всего протестировано пар: {validation_results['total']}
- Прошли валидацию: {validation_results['passed']} ({validation_results['passed']/validation_results['total']*100:.1f}%)
- Не прошли валидацию: {validation_results['failed']} ({validation_results['failed']/validation_results['total']*100:.1f}%)

## 📈 Оценки качества
- Средняя оценка: {validation_results.get('avg_score', 0):.1f}/10
- Минимальная оценка: {validation_results.get('min_score', 0)}/10
- Максимальная оценка: {validation_results.get('max_score', 0)}/10

## 🔧 Основные проблемы
{chr(10).join(f"- {issue}" for issue in set(validation_results['issues']))}

## 📝 Сравнение промптов
### Улучшения в новых промптах:
"""
        
        for improvement in comparison_results['improvements']:
            report += f"""
**Вопрос:** {improvement['question'][:50]}...
- Увеличение длины: +{improvement['length_increase']} символов
- Добавлен контекст эксперта: {'✅' if improvement['has_context'] else '❌'}
- Добавлена специализация: {'✅' if improvement['has_specialization'] else '❌'}
- Технические термины: {'✅' if improvement['has_technical_terms'] else '❌'}
"""
        
        report += f"""
## 🚀 Рекомендации по улучшению

1. **Качество данных**: {validation_results['failed']} пар не прошли валидацию
2. **Промпт-шаблоны**: Новые шаблоны содержат больше контекста
3. **Валидация**: Автоматическая фильтрация низкокачественных данных
4. **Параметры обучения**: Оптимизированы для лучшего качества

## 📊 Заключение
"""
        
        if validation_results.get('avg_score', 0) >= 7:
            report += "✅ Качество данных хорошее, улучшения эффективны"
        elif validation_results.get('avg_score', 0) >= 5:
            report += "⚠️ Качество данных удовлетворительное, есть место для улучшений"
        else:
            report += "❌ Качество данных низкое, требуется дополнительная работа"
        
        return report
    
    def save_results(self, results: Dict, output_path: str):
        """
        Сохраняет результаты тестирования
        
        Args:
            results: Результаты тестирования
            output_path: Путь для сохранения
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Результаты сохранены в: {output_path}")


def main():
    """Основная функция"""
    print("🚀 Запуск тестирования улучшений качества...")
    
    # Загружаем тестовые данные
    test_data_path = "output/base-of-mining-and-mining.final.json"
    
    if not os.path.exists(test_data_path):
        print(f"❌ Файл {test_data_path} не найден")
        return
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_data = data.get('qa_pairs', [])[:20]  # Берем первые 20 пар для тестирования
    
    if not test_data:
        print("❌ Нет данных для тестирования")
        return
    
    print(f"📊 Загружено {len(test_data)} пар для тестирования")
    
    # Создаем тестер
    tester = QualityTester()
    
    # Генерируем отчет
    report = tester.generate_quality_report(test_data)
    
    # Сохраняем отчет
    report_path = "quality_improvements_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 Отчет сохранен в: {report_path}")
    print("\n" + "="*50)
    print(report)
    print("="*50)


if __name__ == "__main__":
    main()


