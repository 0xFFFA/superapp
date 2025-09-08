#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование RAG системы
"""

import os
import sys
import json
from rag_system import MiningRAG, create_ollama_config

def test_ollama_connection():
    """Тестирует подключение к Ollama"""
    print("🔍 Тестирование подключения к Ollama...")
    
    config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    rag = MiningRAG(config)
    
    # Проверяем подключение
    if rag.llm.test_connection():
        print("✅ Подключение к Ollama успешно")
        
        # Получаем список моделей
        models = rag.llm.get_available_models()
        if models:
            print("📋 Доступные модели:")
            for model in models:
                print(f"  - {model['name']} (размер: {model.get('size', 'неизвестно')})")
        else:
            print("⚠️ Не удалось получить список моделей")
        
        return True
    else:
        print("❌ Ошибка подключения к Ollama")
        return False

def test_knowledge_base_loading(knowledge_base_path: str = None):
    """Тестирует загрузку базы знаний"""
    print("\n📚 Тестирование загрузки базы знаний...")
    
    config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    rag = MiningRAG(config)
    
    # Определяем путь к базе знаний
    if knowledge_base_path is None:
        knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'output', 'base-mining-and-mining-quality.final.json')
    
    print(f"📁 Путь к базе знаний: {knowledge_base_path}")
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ Файл базы знаний не найден: {knowledge_base_path}")
        return False
    
    if rag.load_knowledge_base(knowledge_base_path):
        print("✅ База знаний загружена успешно")
        
        # Показываем статистику
        stats = rag.get_stats()
        print(f"📊 Статистика:")
        print(f"  - Пар вопрос-ответ: {stats['total_qa_pairs']}")
        print(f"  - Векторный индекс: {'✅' if stats['vector_index_built'] else '❌'}")
        
        return True
    else:
        print("❌ Ошибка загрузки базы знаний")
        return False

def test_search_functionality():
    """Тестирует функциональность поиска"""
    print("\n🔍 Тестирование поиска...")
    
    config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    rag = MiningRAG(config)
    
    # Загружаем базу знаний
    knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'output', 'base-mining-and-mining-quality.final.json')
    
    if not rag.load_knowledge_base(knowledge_base_path):
        print("❌ Не удалось загрузить базу знаний")
        return False
    
    # Тестовые запросы
    test_queries = [
        "коэффициент крепости",
        "взрывные работы",
        "горные выработки",
        "порода известняк"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Запрос: '{query}'")
        results = rag.vector_store.search(query, top_k=2)
        
        if results:
            print(f"✅ Найдено {len(results)} результатов")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['question'][:50]}... (сходство: {result['similarity']:.3f})")
        else:
            print("❌ Результаты не найдены")
    
    return True

def test_llm_generation():
    """Тестирует генерацию ответов LLM"""
    print("\n🤖 Тестирование генерации ответов...")
    
    config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    rag = MiningRAG(config)
    
    # Простой тест без контекста
    print("🔍 Тест простого запроса...")
    simple_answer = rag.llm.generate("Что такое горное дело?")
    print(f"Ответ: {simple_answer[:100]}...")
    
    # Тест с контекстом
    print("\n🔍 Тест с контекстом...")
    context = "Коэффициент крепости Протодьяконова определяется по формуле f = σсж/100. Для сланцев коэффициент f = 2.5 - 3.0."
    contextual_answer = rag.llm.generate("Как рассчитать коэффициент крепости?", context)
    print(f"Ответ: {contextual_answer[:100]}...")
    
    return True

def run_full_test(knowledge_base_path: str = None):
    """Запускает полный тест системы"""
    print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ RAG СИСТЕМЫ")
    print("=" * 50)
    
    tests = [
        ("Подключение к Ollama", test_ollama_connection),
        ("Загрузка базы знаний", lambda: test_knowledge_base_loading(knowledge_base_path)),
        ("Функциональность поиска", test_search_functionality),
        ("Генерация ответов LLM", test_llm_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоговый отчет
    print(f"\n{'='*50}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ ПРОЙДЕН" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nРезультат: {passed}/{len(results)} тестов пройдено")
    
    if passed == len(results):
        print("🎉 Все тесты пройдены успешно! Система готова к работе.")
    else:
        print("⚠️ Некоторые тесты провалены. Проверьте конфигурацию.")
    
    return passed == len(results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование RAG системы')
    parser.add_argument('--test', choices=['connection', 'knowledge', 'search', 'llm', 'all'], 
                       default='all', help='Какой тест запустить')
    parser.add_argument('--data', '--knowledge-base', 
                       help='Путь к файлу базы знаний (JSON)')
    parser.add_argument('--list-data', action='store_true',
                       help='Показать доступные файлы базы знаний')
    
    args = parser.parse_args()
    
    # Показываем доступные файлы базы знаний
    if args.list_data:
        print("📚 ДОСТУПНЫЕ ФАЙЛЫ БАЗЫ ЗНАНИЙ:")
        print("=" * 40)
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        if os.path.exists(output_dir):
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            for i, file in enumerate(json_files, 1):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"{i}. {file} ({file_size:,} байт)")
        else:
            print("❌ Каталог output не найден")
        return
    
    # Определяем путь к базе знаний
    knowledge_base_path = args.data
    if knowledge_base_path and not os.path.isabs(knowledge_base_path):
        # Если путь относительный, делаем его абсолютным относительно output
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        knowledge_base_path = os.path.join(output_dir, knowledge_base_path)
    
    if args.test == 'connection':
        test_ollama_connection()
    elif args.test == 'knowledge':
        test_knowledge_base_loading(knowledge_base_path)
    elif args.test == 'search':
        test_search_functionality()
    elif args.test == 'llm':
        test_llm_generation()
    elif args.test == 'all':
        run_full_test(knowledge_base_path)
