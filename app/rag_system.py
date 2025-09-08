#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG система для горного дела с использованием Ollama
Дополняет существующую функциональность в каталоге app/
"""

import json
import requests
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import os
import sys
from collections import Counter
from abc import ABC, abstractmethod

# Импорты для семантического поиска
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  sentence-transformers не установлен. Семантический поиск недоступен.")

# Добавляем корневую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class VectorStoreInterface(ABC):
    """Интерфейс для векторных хранилищ"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Добавляет документы в хранилище"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Ищет похожие документы"""
        pass

@dataclass
class OllamaConfig:
    """Конфигурация для Ollama"""
    host: str
    headers: Dict[str, str]
    model: str = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
    timeout: int = 30

class OllamaLLM:
    """LLM клиент для Ollama"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = f"https://{config.host}"
    
    def generate(self, prompt: str, context: str = "", max_tokens: int = 500) -> str:
        """Генерирует ответ с помощью Ollama"""
        # Формируем полный промпт
        if context:
            full_prompt = f"""Контекст из базы знаний по горному делу:
{context}

Вопрос: {prompt}

Ответ (на русском языке, основываясь на контексте):"""
        else:
            full_prompt = f"Вопрос: {prompt}\n\nОтвет (на русском языке):"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.config.headers,
                json={
                    "model": self.config.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": max_tokens
                    }
                },
                timeout=self.config.timeout,
                verify=False  # Отключаем проверку SSL для тестирования
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Ошибка генерации ответа")
            else:
                return f"Ошибка API: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "Ошибка: Превышено время ожидания ответа от Ollama"
        except requests.exceptions.ConnectionError:
            return "Ошибка: Не удалось подключиться к Ollama"
        except Exception as e:
            return f"Ошибка подключения: {e}"
    
    def test_connection(self) -> bool:
        """Тестирует подключение к Ollama"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self.config.headers,
                timeout=5,
                verify=False
            )
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Получает список доступных моделей"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self.config.headers,
                timeout=5,
                verify=False
            )
            if response.status_code == 200:
                return response.json().get('models', [])
            return []
        except:
            return []

class SimpleVectorStore(VectorStoreInterface):
    """Простое векторное хранилище на основе TF-IDF"""
    
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.tf_idf_matrix = []
        self.idf_scores = {}
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Добавляет документы в хранилище"""
        self.documents = documents
        self._build_tfidf_index()
    
    def _build_tfidf_index(self):
        """Строит TF-IDF индекс"""
        if not self.documents:
            return
        
        # Собираем все тексты
        all_texts = []
        for doc in self.documents:
            text = f"{doc['question']} {doc['answer']}"
            words = self._preprocess_text(text)
            all_texts.append(words)
            self.vocabulary.update(words)
        
        self.vocabulary = list(self.vocabulary)
        
        if not self.vocabulary:
            return
        
        # Вычисляем IDF
        total_docs = len(all_texts)
        for word in self.vocabulary:
            doc_count = sum(1 for text in all_texts if word in text)
            self.idf_scores[word] = np.log(total_docs / doc_count) if doc_count > 0 else 0
        
        # Вычисляем TF-IDF для каждого документа
        for text in all_texts:
            tf_scores = Counter(text)
            tf_idf_vector = []
            
            for word in self.vocabulary:
                tf = tf_scores.get(word, 0) / len(text) if len(text) > 0 else 0
                idf = self.idf_scores[word]
                tf_idf_vector.append(tf * idf)
            
            self.tf_idf_matrix.append(tf_idf_vector)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [word for word in words if len(word) >= 3]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Ищет похожие документы"""
        if not self.documents or not self.tf_idf_matrix:
            return []
        
        # Предобрабатываем запрос
        query_words = self._preprocess_text(query)
        if not query_words:
            return []
        
        # Создаем TF-IDF вектор для запроса
        query_tf = Counter(query_words)
        query_vector = []
        
        for word in self.vocabulary:
            tf = query_tf.get(word, 0) / len(query_words)
            idf = self.idf_scores.get(word, 0)
            query_vector.append(tf * idf)
        
        # Вычисляем сходство
        similarities = []
        for i, doc_vector in enumerate(self.tf_idf_matrix):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K результатов
        results = []
        for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
            if similarity > 0:
                doc = self.documents[doc_idx]
                results.append({
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'similarity': similarity,
                    'rank': i + 1
                })
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисляет косинусное сходство"""
        if len(vec1) != len(vec2):
            return 0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = np.sqrt(sum(a * a for a in vec1))
        magnitude2 = np.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)

class SemanticVectorStore(VectorStoreInterface):
    """Семантическое векторное хранилище на основе эмбеддингов"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not SEMANTIC_AVAILABLE:
            raise ImportError("sentence-transformers не установлен. Установите: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Добавляет документы в хранилище"""
        self.documents = documents
        
        # Создаем тексты для эмбеддингов
        texts = []
        for doc in documents:
            text = f"{doc['question']} {doc['answer']}"
            texts.append(text)
        
        # Генерируем эмбеддинги
        print("🔄 Генерация эмбеддингов...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✅ Создано {len(self.embeddings)} эмбеддингов")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Ищет похожие документы"""
        if not self.documents or self.embeddings is None:
            return []
        
        # Генерируем эмбеддинг для запроса
        query_embedding = self.model.encode([query])
        
        # Вычисляем сходство
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding[0], doc_embedding)
            similarities.append((i, similarity))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K результатов
        results = []
        for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
            if similarity > 0:
                doc = self.documents[doc_idx]
                results.append({
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'similarity': float(similarity),
                    'rank': i + 1
                })
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисляет косинусное сходство"""
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

class HybridVectorStore(VectorStoreInterface):
    """Гибридное векторное хранилище (TF-IDF + семантический поиск)"""
    
    def __init__(self, semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tfidf_store = SimpleVectorStore()
        self.semantic_store = None
        
        if SEMANTIC_AVAILABLE:
            self.semantic_store = SemanticVectorStore(semantic_model)
        else:
            print("⚠️  Семантический поиск недоступен, используется только TF-IDF")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Добавляет документы в оба хранилища"""
        # Добавляем в TF-IDF
        self.tfidf_store.add_documents(documents)
        
        # Добавляем в семантическое хранилище (если доступно)
        if self.semantic_store:
            self.semantic_store.add_documents(documents)
    
    def search(self, query: str, top_k: int = 3, tfidf_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Гибридный поиск с весами для TF-IDF и семантического поиска"""
        results = []
        
        # TF-IDF поиск
        tfidf_results = self.tfidf_store.search(query, top_k * 2)
        
        # Семантический поиск (если доступен)
        if self.semantic_store:
            semantic_results = self.semantic_store.search(query, top_k * 2)
            
            # Объединяем результаты с весами
            combined_scores = {}
            
            for result in tfidf_results:
                key = f"{result['question']}|{result['answer']}"
                combined_scores[key] = {
                    'question': result['question'],
                    'answer': result['answer'],
                    'tfidf_score': result['similarity'],
                    'semantic_score': 0.0,
                    'combined_score': result['similarity'] * tfidf_weight
                }
            
            for result in semantic_results:
                key = f"{result['question']}|{result['answer']}"
                if key in combined_scores:
                    combined_scores[key]['semantic_score'] = result['similarity']
                    combined_scores[key]['combined_score'] += result['similarity'] * semantic_weight
                else:
                    combined_scores[key] = {
                        'question': result['question'],
                        'answer': result['answer'],
                        'tfidf_score': 0.0,
                        'semantic_score': result['similarity'],
                        'combined_score': result['similarity'] * semantic_weight
                    }
            
            # Сортируем по комбинированному скору
            sorted_results = sorted(combined_scores.values(), key=lambda x: x['combined_score'], reverse=True)
            
            # Формируем финальные результаты
            for i, result in enumerate(sorted_results[:top_k]):
                if result['combined_score'] > 0:
                    results.append({
                        'question': result['question'],
                        'answer': result['answer'],
                        'similarity': result['combined_score'],
                        'tfidf_score': result['tfidf_score'],
                        'semantic_score': result['semantic_score'],
                        'rank': i + 1
                    })
        else:
            # Только TF-IDF, если семантический поиск недоступен
            results = tfidf_results[:top_k]
        
        return results

class MiningRAG:
    """RAG система для горного дела"""
    
    def __init__(self, ollama_config: OllamaConfig, search_type: str = "tfidf"):
        self.llm = OllamaLLM(ollama_config)
        self.search_type = search_type
        self.qa_pairs = []
        self.knowledge_base_path = None
        
        # Выбираем тип векторного хранилища
        if search_type == "tfidf":
            self.vector_store = SimpleVectorStore()
        elif search_type == "semantic":
            if not SEMANTIC_AVAILABLE:
                print("⚠️  Семантический поиск недоступен, используется TF-IDF")
                self.vector_store = SimpleVectorStore()
                self.search_type = "tfidf"
            else:
                self.vector_store = SemanticVectorStore()
        elif search_type == "hybrid":
            self.vector_store = HybridVectorStore()
        else:
            raise ValueError(f"Неизвестный тип поиска: {search_type}. Доступные: tfidf, semantic, hybrid")
    
    def load_knowledge_base(self, json_file_path: str):
        """Загружает базу знаний из JSON файла"""
        print("📚 Загрузка базы знаний...")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.qa_pairs = data['qa_pairs']
            self.knowledge_base_path = json_file_path
            print(f"✅ Загружено {len(self.qa_pairs)} пар вопрос-ответ")
            
            # Строим векторный индекс
            self.vector_store.add_documents(self.qa_pairs)
            print("✅ Векторный индекс построен")
            
        except FileNotFoundError:
            print(f"❌ Файл не найден: {json_file_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False
        
        return True
    
    def ask_question(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """Задает вопрос системе"""
        if not question.strip():
            return {
                "answer": "Пожалуйста, задайте вопрос.",
                "sources": [],
                "confidence": 0.0,
                "method": "error"
            }
        
        if use_rag and self.qa_pairs:
            # Ищем релевантные ответы
            search_results = self.vector_store.search(question, top_k=3)
            
            if search_results:
                # Формируем контекст
                context = "\n\n".join([
                    f"Вопрос: {result['question']}\nОтвет: {result['answer']}"
                    for result in search_results
                ])
                
                # Генерируем ответ с помощью LLM
                answer = self.llm.generate(question, context)
                
                return {
                    "answer": answer,
                    "sources": search_results,
                    "confidence": max(result['similarity'] for result in search_results),
                    "method": f"rag_{self.search_type}"
                }
            else:
                # Если не нашли релевантных результатов, используем LLM без контекста
                answer = self.llm.generate(question)
                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.0,
                    "method": "llm_only"
                }
        else:
            # Прямой поиск без LLM
            search_results = self.vector_store.search(question, top_k=1)
            if search_results:
                return {
                    "answer": search_results[0]['answer'],
                    "sources": search_results,
                    "confidence": search_results[0]['similarity'],
                    "method": "direct_search"
                }
            else:
                return {
                    "answer": "Информация не найдена в базе знаний.",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "not_found"
                }
    
    def test_system(self) -> bool:
        """Тестирует систему"""
        print("🔍 Тестирование системы...")
        
        # Тест подключения к Ollama
        if self.llm.test_connection():
            print("✅ Подключение к Ollama успешно")
        else:
            print("❌ Ошибка подключения к Ollama")
            return False
        
        # Тест загрузки базы знаний
        if self.qa_pairs:
            print(f"✅ База знаний загружена: {len(self.qa_pairs)} пар")
        else:
            print("❌ База знаний не загружена")
            return False
        
        # Тест поиска
        test_question = "Что такое коэффициент крепости?"
        search_results = self.vector_store.search(test_question, top_k=1)
        if search_results:
            print(f"✅ Поиск работает: найдено {len(search_results)} результатов")
        else:
            print("❌ Ошибка поиска")
            return False
        
        print("✅ Все тесты пройдены успешно")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику системы"""
        return {
            "knowledge_base_loaded": len(self.qa_pairs) > 0,
            "total_qa_pairs": len(self.qa_pairs),
            "ollama_connected": self.llm.test_connection(),
            "knowledge_base_path": self.knowledge_base_path,
            "vector_index_built": len(self.vector_store.tf_idf_matrix) > 0
        }

def create_ollama_config(host: str, token: str, model: str = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest") -> OllamaConfig:
    """Создает конфигурацию Ollama"""
    return OllamaConfig(
        host=host,
        headers={'X-Access-Token': token},
        model=model
    )

def demo_rag_system(knowledge_base_path: str = None, search_type: str = "tfidf"):
    """Демонстрация работы RAG системы"""
    print("🏭 ДЕМОНСТРАЦИЯ RAG СИСТЕМЫ ДЛЯ ГОРНОГО ДЕЛА")
    print("=" * 60)
    print(f"🔍 Тип поиска: {search_type.upper()}")
    print("=" * 60)
    
    # Конфигурация Ollama
    ollama_config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    # Создаем RAG систему
    rag = MiningRAG(ollama_config, search_type=search_type)
    
    # Проверяем доступные модели
    print("🔍 Проверка доступных моделей...")
    models = rag.llm.get_available_models()
    if models:
        print("Доступные модели:")
        for model in models:
            print(f"  - {model['name']}")
    else:
        print("❌ Не удалось получить список моделей")
        return
    
    # Определяем путь к базе знаний
    if knowledge_base_path is None:
        knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'output', 'base-mining-and-mining-quality.final.json')
    
    print(f"📚 Загружаем базу знаний из: {knowledge_base_path}")
    
    if not rag.load_knowledge_base(knowledge_base_path):
        print("❌ Не удалось загрузить базу знаний")
        return
    
    # Тестируем систему
    if not rag.test_system():
        print("❌ Система не готова к работе")
        return
    
    # Примеры вопросов
    test_questions = [
        "Как рассчитать коэффициент крепости Протодьяконова?",
        "Какая минимальная толщина выработки для взрывных работ?",
        "Как определить линию падения пласта?",
        "Что такое коэффициент сжимаемости породы?",
        "Как выбрать способ крепления для горных выработок?"
    ]
    
    print("\n❓ ТЕСТИРОВАНИЕ ВОПРОСОВ:")
    print("-" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("🤔 Думаю...")
        
        # Получаем ответ
        result = rag.ask_question(question)
        
        print(f"💡 Ответ: {result['answer']}")
        print(f"📊 Метод: {result['method']}")
        print(f"🎯 Уверенность: {result['confidence']:.2f}")
        
        if result['sources']:
            print(f"📚 Источники: {len(result['sources'])}")
            for j, source in enumerate(result['sources'][:2], 1):
                print(f"  {j}. {source['question'][:60]}... (сходство: {source['similarity']:.2f})")
        
        print("-" * 60)
    
    print("\n✅ Демонстрация завершена!")

def interactive_mode(knowledge_base_path: str = None, search_type: str = "tfidf"):
    """Интерактивный режим для задавания вопросов"""
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print(f"🔍 Тип поиска: {search_type.upper()}")
    print("Введите 'выход' для завершения")
    print("=" * 40)
    
    # Конфигурация Ollama
    ollama_config = create_ollama_config(
        host='193.247.73.14:11436',
        token='k6Svw7EldnQLhBpivenz7E2Z01H8FF',
        model='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest'
    )
    
    # Создаем RAG систему
    rag = MiningRAG(ollama_config, search_type=search_type)
    
    # Определяем путь к базе знаний
    if knowledge_base_path is None:
        knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'output', 'base-mining-and-mining-quality.final.json')
    
    print(f"📚 Загружаем базу знаний из: {knowledge_base_path}")
    
    if not rag.load_knowledge_base(knowledge_base_path):
        print("❌ Не удалось загрузить базу знаний")
        return
    
    print("✅ Система готова к работе!")
    
    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()
            
            if question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("👋 До свидания!")
                break
            
            if not question:
                continue
            
            print("🤔 Думаю...")
            result = rag.ask_question(question)
            
            print(f"\n💡 Ответ: {result['answer']}")
            print(f"📊 Уверенность: {result['confidence']:.2f}")
            print(f"🔍 Метод: {result['method']}")
            
            if result['sources']:
                print(f"📚 Найдено источников: {len(result['sources'])}")
        
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG система для горного дела')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='demo',
                       help='Режим работы: demo или interactive')
    parser.add_argument('--model', default='yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest',
                       help='Модель Ollama для использования')
    parser.add_argument('--data', '--knowledge-base', 
                       help='Путь к файлу базы знаний (JSON)')
    parser.add_argument('--list-data', action='store_true',
                       help='Показать доступные файлы базы знаний')
    parser.add_argument('--search-type', choices=['tfidf', 'semantic', 'hybrid'], 
                       default='tfidf', help='Тип поиска: tfidf, semantic, hybrid')
    
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
        exit(0)
    
    # Определяем путь к базе знаний
    knowledge_base_path = args.data
    if knowledge_base_path and not os.path.isabs(knowledge_base_path):
        # Если путь относительный, делаем его абсолютным относительно output
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        knowledge_base_path = os.path.join(output_dir, knowledge_base_path)
    
    if args.mode == 'demo':
        demo_rag_system(knowledge_base_path, args.search_type)
    elif args.mode == 'interactive':
        interactive_mode(knowledge_base_path, args.search_type)
