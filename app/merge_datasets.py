#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для объединения .json и .mid файлов
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict


class DatasetMerger:
    """Объединитель датасетов"""
    
    def __init__(self):
        self.existing_qa_pairs = []
        self.new_qa_pairs = []
    
    def load_json_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """
        Загружает датасет из .json файла
        
        Args:
            file_path: Путь к .json файлу
            
        Returns:
            Список пар вопрос-ответ
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлекаем пары вопрос-ответ
            if 'qa_pairs' in data:
                qa_pairs = data['qa_pairs']
            elif isinstance(data, list):
                qa_pairs = data
            else:
                print(f"⚠️  Неизвестная структура в {file_path}")
                return []
            
            print(f"✅ Загружено {len(qa_pairs)} пар из {file_path}")
            return qa_pairs
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке {file_path}: {e}")
            return []
    
    def load_mid_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """
        Загружает датасет из .mid файла
        
        Args:
            file_path: Путь к .mid файлу
            
        Returns:
            Список пар вопрос-ответ
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Извлекаем пары вопрос-ответ
            if 'qa_pairs' in data:
                qa_pairs = data['qa_pairs']
            elif isinstance(data, list):
                qa_pairs = data
            else:
                print(f"⚠️  Неизвестная структура в {file_path}")
                return []
            
            print(f"✅ Загружено {len(qa_pairs)} пар из {file_path}")
            return qa_pairs
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке {file_path}: {e}")
            return []
    
    def merge_datasets(self, json_file: str, mid_file: str) -> List[Dict[str, str]]:
        """
        Объединяет два датасета
        
        Args:
            json_file: Путь к .json файлу
            mid_file: Путь к .mid файлу
            
        Returns:
            Объединенный список пар вопрос-ответ
        """
        # Загружаем существующий датасет
        self.existing_qa_pairs = self.load_json_dataset(json_file)
        
        # Загружаем новый датасет
        self.new_qa_pairs = self.load_mid_dataset(mid_file)
        
        # Объединяем
        merged_pairs = self.existing_qa_pairs + self.new_qa_pairs
        
        print(f"📊 Объединено: {len(self.existing_qa_pairs)} + {len(self.new_qa_pairs)} = {len(merged_pairs)} пар")
        
        return merged_pairs
    
    def save_merged_dataset(self, merged_pairs: List[Dict[str, str]], output_path: str, 
                           original_json_path: str) -> bool:
        """
        Сохраняет объединенный датасет
        
        Args:
            merged_pairs: Объединенный список пар
            output_path: Путь для сохранения
            original_json_path: Путь к исходному .json файлу
            
        Returns:
            True если успешно, False в противном случае
        """
        try:
            # Создаем структуру выходного файла
            output_data = {
                "source_pdf": "merged_dataset",
                "total_questions": len(merged_pairs),
                "generation_date": self.get_current_time(),
                "model_used": "merged_from_multiple_sources",
                "original_json": original_json_path,
                "qa_pairs": merged_pairs
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Сохранено {len(merged_pairs)} пар в {output_path}")
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
        description="Объединитель датасетов .json и .mid"
    )
    parser.add_argument("json_file", help="Путь к .json файлу")
    parser.add_argument("mid_file", help="Путь к .mid файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения объединенного файла")
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    if not Path(args.json_file).exists():
        print(f"❌ Файл не найден: {args.json_file}")
        sys.exit(1)
    
    if not Path(args.mid_file).exists():
        print(f"❌ Файл не найден: {args.mid_file}")
        sys.exit(1)
    
    # Определяем путь для сохранения
    if args.output:
        output_path = args.output
    else:
        # Создаем merged файл в том же каталоге, где находятся исходные файлы
        json_path = Path(args.json_file)
        base_name = json_path.stem
        output_path = json_path.parent / f"{base_name}.final.json"
    
    print("🔗 Объединяю датасеты...")
    print(f"📁 JSON файл: {args.json_file}")
    print(f"📁 MID файл: {args.mid_file}")
    print(f"💾 Выходной файл: {output_path}")
    print("-" * 50)
    
    # Создаем объединитель
    merger = DatasetMerger()
    
    # Объединяем датасеты
    merged_pairs = merger.merge_datasets(args.json_file, args.mid_file)
    
    if merged_pairs:
        # Сохраняем результат
        if merger.save_merged_dataset(merged_pairs, output_path, args.json_file):
            print(f"✅ Успешно объединен датасет: {output_path}")
        else:
            print("❌ Не удалось сохранить объединенный датасет")
            sys.exit(1)
    else:
        print("❌ Не удалось объединить датасеты")
        sys.exit(1)


if __name__ == "__main__":
    main()
