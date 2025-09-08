#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый запуск RAG системы
"""

import os
import sys
from rag_system import demo_rag_system, interactive_mode

def main():
    """Главная функция"""
    print("🏭 RAG СИСТЕМА ДЛЯ ГОРНОГО ДЕЛА")
    print("=" * 40)
    print("1. Демонстрация (demo)")
    print("2. Интерактивный режим (interactive)")
    print("3. Тестирование (test)")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nВыберите режим (1-3) или 'q' для выхода: ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("👋 До свидания!")
                break
            elif choice == '1':
                print("\n🚀 Запуск демонстрации...")
                demo_rag_system()
            elif choice == '2':
                print("\n💬 Запуск интерактивного режима...")
                interactive_mode()
            elif choice == '3':
                print("\n🧪 Запуск тестирования...")
                os.system(f"{sys.executable} test_rag.py")
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
        
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
