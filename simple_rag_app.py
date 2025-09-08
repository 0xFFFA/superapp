#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простое RAG приложение для базы знаний по горному делу
Использует TF-IDF и косинусное сходство для поиска релевантных ответов
"""

import json
import math
import re
from collections import Counter
from flask import Flask, render_template, request, jsonify
import os

class SimpleRAG:
    """Простой RAG с TF-IDF векторизацией"""
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.qa_pairs = []
        self.vocabulary = set()
        self.tf_idf_matrix = []
        self.idf_scores = {}
        self.load_data()
        self.build_index()
    
    def load_data(self):
        """Загружает данные из JSON файла"""
        print("📚 Загрузка базы знаний...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.qa_pairs = data['qa_pairs']
        print(f"✅ Загружено {len(self.qa_pairs)} пар вопрос-ответ")
    
    def preprocess_text(self, text):
        """Предобработка текста: токенизация и нормализация"""
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем знаки препинания, оставляем только буквы и цифры
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Разбиваем на слова
        words = text.split()
        
        # Удаляем короткие слова (меньше 3 символов)
        words = [word for word in words if len(word) >= 3]
        
        return words
    
    def build_index(self):
        """Строит TF-IDF индекс для поиска"""
        print("🔍 Построение поискового индекса...")
        
        # Собираем все тексты для построения словаря
        all_texts = []
        for qa in self.qa_pairs:
            # Объединяем вопрос и ответ для поиска
            combined_text = f"{qa['question']} {qa['answer']}"
            processed_text = self.preprocess_text(combined_text)
            all_texts.append(processed_text)
            
            # Добавляем слова в словарь
            self.vocabulary.update(processed_text)
        
        self.vocabulary = list(self.vocabulary)
        print(f"📝 Словарь содержит {len(self.vocabulary)} уникальных слов")
        
        # Вычисляем IDF для каждого слова
        total_docs = len(all_texts)
        for word in self.vocabulary:
            # Количество документов, содержащих это слово
            doc_count = sum(1 for text in all_texts if word in text)
            self.idf_scores[word] = math.log(total_docs / doc_count) if doc_count > 0 else 0
        
        # Вычисляем TF-IDF для каждого документа
        for text in all_texts:
            tf_scores = Counter(text)
            tf_idf_vector = []
            
            for word in self.vocabulary:
                tf = tf_scores.get(word, 0) / len(text) if len(text) > 0 else 0
                idf = self.idf_scores[word]
                tf_idf_vector.append(tf * idf)
            
            self.tf_idf_matrix.append(tf_idf_vector)
        
        print("✅ Индекс построен успешно")
    
    def cosine_similarity(self, vec1, vec2):
        """Вычисляет косинусное сходство между двумя векторами"""
        if len(vec1) != len(vec2):
            return 0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query, top_k=5):
        """Ищет наиболее релевантные ответы на запрос"""
        # Предобрабатываем запрос
        processed_query = self.preprocess_text(query)
        
        # Создаем TF-IDF вектор для запроса
        query_tf = Counter(processed_query)
        query_vector = []
        
        for word in self.vocabulary:
            tf = query_tf.get(word, 0) / len(processed_query) if len(processed_query) > 0 else 0
            idf = self.idf_scores.get(word, 0)
            query_vector.append(tf * idf)
        
        # Вычисляем сходство с каждым документом
        similarities = []
        for i, doc_vector in enumerate(self.tf_idf_matrix):
            similarity = self.cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K результатов
        results = []
        for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
            if similarity > 0:  # Только релевантные результаты
                qa = self.qa_pairs[doc_idx]
                results.append({
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'similarity': similarity,
                    'rank': i + 1
                })
        
        return results
    
    def generate_answer(self, query, context_limit=3):
        """Генерирует ответ на основе найденного контекста"""
        # Ищем релевантные ответы
        search_results = self.search(query, top_k=context_limit)
        
        if not search_results:
            return {
                'answer': "Извините, я не нашел релевантной информации в базе знаний.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Формируем ответ на основе найденных источников
        sources = []
        answer_parts = []
        
        for result in search_results:
            sources.append({
                'question': result['question'],
                'answer': result['answer'],
                'similarity': result['similarity']
            })
            
            # Если сходство высокое, добавляем ответ
            if result['similarity'] > 0.1:
                answer_parts.append(result['answer'])
        
        # Объединяем найденные ответы
        if answer_parts:
            combined_answer = " ".join(answer_parts[:2])  # Берем максимум 2 источника
            confidence = max(result['similarity'] for result in search_results)
        else:
            combined_answer = search_results[0]['answer']
            confidence = search_results[0]['similarity']
        
        return {
            'answer': combined_answer,
            'sources': sources,
            'confidence': confidence
        }

# Создаем Flask приложение
app = Flask(__name__)

# Инициализируем RAG систему
rag_system = SimpleRAG('output/base-mining-and-mining-quality.final.json')

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """API для поиска ответов"""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Пустой запрос'}), 400
    
    # Генерируем ответ
    result = rag_system.generate_answer(query)
    
    return jsonify(result)

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Возвращает примеры вопросов для демонстрации"""
    suggestions = [
        "Как рассчитать коэффициент крепости Протодьяконова?",
        "Какая минимальная толщина выработки для взрывных работ?",
        "Как определить линию падения пласта?",
        "Как выбрать способ крепления для горных выработок?",
        "Как рассчитать нагрузку на шнековую линию?",
        "Что такое коэффициент сжимаемости породы?",
        "Как определить предел прочности породы на трение?",
        "Как вычислить угол падения при горных работах?"
    ]
    
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    print("🚀 Запуск RAG приложения...")
    print("📚 База знаний: 326 пар вопрос-ответ по горному делу")
    print("🌐 Веб-интерфейс: http://localhost:5000")
    
    # Создаем папку для шаблонов
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
