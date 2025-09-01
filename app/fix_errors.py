#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа .err файла и создания .mid файла с исправленными данными
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional


class ErrorFileAnalyzer:
    """Анализатор .err файлов для извлечения пар вопрос-ответ"""
    
    def __init__(self):
        self.qa_pairs = []
    
    def extract_qa_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Извлекает пары вопрос-ответ из текста ответа модели
        
        Args:
            text: Текст ответа модели
            
        Returns:
            Список словарей с вопросами и ответами
        """
        pairs = []
        
        # Убираем <think> блоки если есть
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Простой и надежный паттерн для поиска всех пар question-answer
        pattern = r'"question"\s*:\s*"([^"]+)"[^}]*"answer"\s*:\s*"([^"]+)"'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            if len(match) >= 2:
                question = match[0].strip()
                answer = match[1].strip()
                
                # Очищаем от лишних символов
                question = self.clean_text(question)
                answer = self.clean_text(answer)
                
                if question and answer and len(question) > 5 and len(answer) > 5:
                    pairs.append({
                        "question": question,
                        "answer": answer
                    })
        
        return pairs
    
    def clean_text(self, text: str) -> str:
        """
        Очищает текст от лишних символов
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Убираем лишние escape-символы
        text = text.replace('\\"', '"')
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\r', ' ')
        
        # Убираем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем лишние символы в начале и конце
        text = text.strip()
        
        return text
    
    def analyze_error_file(self, error_file_path: str) -> List[Dict[str, str]]:
        """
        Анализирует .err файл и извлекает все пары вопрос-ответ
        
        Args:
            error_file_path: Путь к .err файлу
            
        Returns:
            Список всех найденных пар вопрос-ответ
        """
        try:
            with open(error_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разбиваем на блоки по разделителю
            blocks = content.split('=' * 50)
            
            for block in blocks:
                if 'Ответ модели:' in block:
                    # Извлекаем ответ модели
                    response_start = block.find('Ответ модели:')
                    if response_start != -1:
                        response_text = block[response_start + len('Ответ модели:'):].strip()
                        
                        # Ищем пары вопрос-ответ
                        pairs = self.extract_qa_from_text(response_text)
                        if pairs:
                            print(f"🔍 В блоке найдено {len(pairs)} пар")
                            self.qa_pairs.extend(pairs)
                        else:
                            print(f"⚠️  В блоке не найдено пар")
            
            return self.qa_pairs
            
        except Exception as e:
            print(f"❌ Ошибка при анализе файла {error_file_path}: {e}")
            return []
    
    def save_to_mid_file(self, output_path: str) -> bool:
        """
        Сохраняет извлеченные пары в .mid файл
        
        Args:
            output_path: Путь для сохранения .mid файла
            
        Returns:
            True если успешно, False в противном случае
        """
        try:
            if not self.qa_pairs:
                print("⚠️  Нет данных для сохранения")
                return False
            
            output_data = {
                "source_error_file": "extracted_from_err",
                "total_questions": len(self.qa_pairs),
                "extraction_date": self.get_current_time(),
                "qa_pairs": self.qa_pairs
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Сохранено {len(self.qa_pairs)} пар в {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
            return False
    
    def get_current_time(self) -> str:
        """Возвращает текущее время в строковом формате"""
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Основная функция программы"""
    parser = argparse.ArgumentParser(
        description="Анализатор .err файлов для извлечения пар вопрос-ответ"
    )
    parser.add_argument("error_file", help="Путь к .err файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения .mid файла")
    
    args = parser.parse_args()
    
    # Проверяем существование .err файла
    if not Path(args.error_file).exists():
        print(f"❌ Файл не найден: {args.error_file}")
        sys.exit(1)
    
    # Определяем путь для сохранения
    if args.output:
        output_path = args.output
    else:
        # Создаем .mid файл в том же каталоге, где находится .err файл
        error_path = Path(args.error_file)
        error_name = error_path.stem
        output_path = error_path.parent / f"{error_name}.mid"
    
    print("🔍 Анализирую .err файл...")
    print(f"📁 Файл ошибок: {args.error_file}")
    print(f"💾 Выходной файл: {output_path}")
    print("-" * 50)
    
    # Создаем анализатор
    analyzer = ErrorFileAnalyzer()
    
    # Анализируем .err файл
    qa_pairs = analyzer.analyze_error_file(args.error_file)
    
    if qa_pairs:
        print(f"📊 Найдено {len(qa_pairs)} пар вопрос-ответ")
        
        # Сохраняем в .mid файл
        if analyzer.save_to_mid_file(output_path):
            print(f"✅ Успешно создан .mid файл: {output_path}")
        else:
            print("❌ Не удалось создать .mid файл")
            sys.exit(1)
    else:
        print("⚠️  Не найдено ни одной пары вопрос-ответ")
        sys.exit(1)


if __name__ == "__main__":
    main()
