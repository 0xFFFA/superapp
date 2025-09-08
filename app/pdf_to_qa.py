#!/usr/bin/env python3
"""
PDF to Q&A Dataset Generator
Генератор датасета вопросов-ответов из PDF файлов с использованием Ollama
Улучшенная версия с валидацией качества данных
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import requests
import time
import urllib3

# Подавляем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Ошибка: Необходимо установить PyPDF2 и pdfplumber")
    print("Выполните: pip install PyPDF2 pdfplumber")
    sys.exit(1)

# Импортируем улучшенные промпт-шаблоны
from prompt_templates import PromptTemplates, validate_qa_quality


class PDFToQAGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.1:8b", api_token: str = None):
        """
        Инициализация генератора
        
        Args:
            ollama_url: URL для Ollama API
            model: Модель Ollama для генерации
            api_token: Токен аутентификации для Ollama
        """
        self.ollama_url = ollama_url
        self.model = model
        self.api_token = api_token
        self.max_chunk_size = 4000  # Максимальный размер текстового блока для Ollama
        
    def read_pdf(self, pdf_path: str) -> str:
        """
        Читает PDF файл и извлекает текст
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Извлеченный текст
        """
        print(f"Читаю PDF файл: {pdf_path}")
        
        try:
            # Пробуем сначала pdfplumber для лучшего качества
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            if not text.strip():
                # Fallback к PyPDF2 если pdfplumber не сработал
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                        
            if not text.strip():
                raise ValueError("Не удалось извлечь текст из PDF")
                
            print(f"Успешно извлечено {len(text)} символов")
            return text.strip()
            
        except Exception as e:
            print(f"Ошибка при чтении PDF: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на логические блоки для обработки
        
        Args:
            text: Исходный текст
            
        Returns:
            Список текстовых блоков
        """
        print("Разбиваю текст на блоки...")
        
        # Разбиваем по абзацам
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Если текущий блок + новый абзац превышает лимит
            if len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Если один абзац слишком длинный, разбиваем его
                    if len(paragraph) > self.max_chunk_size:
                        words = paragraph.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= self.max_chunk_size:
                                temp_chunk += word + " "
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word + " "
                        current_chunk = temp_chunk
                    else:
                        current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Создано {len(chunks)} текстовых блоков")
        return chunks
    

    

    
    def generate_qa_pairs(self, text_chunk: str, num_questions: int, error_log_path: str = None) -> List[Dict[str, str]]:
        """
        Генерирует пары вопрос-ответ для текстового блока
        
        Args:
            text_chunk: Текстовый блок
            num_questions: Количество вопросов
            error_log_path: Путь к файлу для логирования ошибок
            
        Returns:
            Список словарей с вопросами и ответами
        """
        return self._make_ollama_request(text_chunk, num_questions, error_log_path)
    

    
    def _make_ollama_request(self, text_chunk: str, num_questions: int, error_log_path: str = None) -> List[Dict[str, str]]:
        """
        Выполняет запрос к Ollama API с улучшенным промптом
        
        Args:
            text_chunk: Текстовый блок
            num_questions: Количество вопросов
            
        Returns:
            Список словарей с вопросами и ответами
        """
        # Используем улучшенный промпт из PromptTemplates
        prompt = PromptTemplates.get_qa_generation_prompt(text_chunk, num_questions)
        
        # Адаптивный таймаут в зависимости от размера блока и количества вопросов
        base_timeout = 120
        size_factor = len(text_chunk) // 1000 * 30  # +30 сек на каждые 1000 символов
        questions_factor = num_questions * 15  # +15 сек на каждый вопрос
        adaptive_timeout = min(600, base_timeout + size_factor + questions_factor)
        
        # Подготавливаем заголовки
        headers = {}
        if self.api_token:
            headers['X-Access-Token'] = self.api_token
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            headers=headers,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=adaptive_timeout,
            verify=False  # Отключаем проверку SSL для самоподписанных сертификатов
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            # Пытаемся распарсить JSON
            try:
                # Очищаем ответ от markdown блоков если они есть
                cleaned_response = response_text.strip()
                if cleaned_response.startswith('```json'):
                    # Убираем markdown блоки
                    cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
                elif cleaned_response.startswith('```'):
                    # Убираем общие markdown блоки
                    cleaned_response = cleaned_response.replace('```', '').strip()
                
                parsed_data = json.loads(cleaned_response)
                print(f"🔍 Тип ответа: {type(parsed_data)}")
                
                # Обрабатываем случай с полем "questions"
                if isinstance(parsed_data, dict):
                    if "questions" in parsed_data:
                        qa_pairs = parsed_data["questions"]
                        print(f"✅ Найдено поле 'questions' с {len(qa_pairs)} элементами")
                    else:
                        print(f"⚠️  Словарь без поля 'questions', ключи: {list(parsed_data.keys())}")
                        if error_log_path:
                            with open(error_log_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n{'='*50}\n")
                                f.write(f"Словарь без поля 'questions'\n")
                                f.write(f"Ключи: {list(parsed_data.keys())}\n")
                                f.write(f"Ответ модели:\n{response_text}\n")
                        return []
                else:
                    qa_pairs = parsed_data
                    print(f"✅ Прямой массив с {len(qa_pairs)} элементами")
                
                # Проверяем структуру
                if isinstance(qa_pairs, list) and all(
                    isinstance(qa, dict) and 'question' in qa and 'answer' in qa 
                    for qa in qa_pairs
                ):
                    # Валидируем качество каждой пары вопрос-ответ
                    validated_pairs = []
                    for qa in qa_pairs:
                        validation = validate_qa_quality(qa['question'], qa['answer'])
                        if validation['passed']:
                            validated_pairs.append(qa)
                        else:
                            print(f"⚠️  Пара отфильтрована (оценка {validation['quality_score']}/10): {qa['question'][:50]}...")
                            if error_log_path:
                                self._log_quality_issue(error_log_path, qa, validation)
                    
                    print(f"✅ Успешно извлечено {len(qa_pairs)} пар, {len(validated_pairs)} прошли валидацию качества")
                    return validated_pairs
                else:
                    print("⚠️  Неверная структура данных")
                    if error_log_path:
                        with open(error_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"\n{'='*50}\n")
                            f.write(f"Неверная структура данных\n")
                            f.write(f"Тип: {type(qa_pairs)}\n")
                            f.write(f"Данные: {qa_pairs}\n")
                            f.write(f"Ответ модели:\n{response_text}\n")
                    return []
                    
            except json.JSONDecodeError as e:
                print("⚠️  Ошибка парсинга JSON")
                if error_log_path:
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*50}\n")
                        f.write(f"Ошибка парсинга JSON: {e}\n")
                        f.write(f"Ответ модели:\n{response_text}\n")
                return []
        else:
            print(f"Ошибка API Ollama: {response.status_code}")
            return []
    
    def _log_quality_issue(self, error_log_path: str, qa_pair: Dict, validation: Dict):
        """
        Логирует проблемы качества данных
        
        Args:
            error_log_path: Путь к файлу логов
            qa_pair: Пара вопрос-ответ
            validation: Результат валидации
        """
        try:
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"ПРОБЛЕМА КАЧЕСТВА ДАННЫХ\n")
                f.write(f"Оценка: {validation['quality_score']}/10\n")
                f.write(f"Проблемы: {', '.join(validation['issues'])}\n")
                f.write(f"Вопрос: {qa_pair['question']}\n")
                f.write(f"Ответ: {qa_pair['answer']}\n")
        except Exception as e:
            print(f"Ошибка записи в лог: {e}")
    
    def process_pdf(self, pdf_path: str, output_path: str, questions_per_chunk: int = 3) -> bool:
        """
        Основной метод для обработки PDF и создания датасета
        
        Args:
            pdf_path: Путь к PDF файлу
            output_path: Путь для сохранения результата
            questions_per_chunk: Количество вопросов на текстовый блок
            
        Returns:
            True если успешно, False в противном случае
        """
        try:
            # Читаем PDF
            text = self.read_pdf(pdf_path)
            
            # Разбиваем на блоки
            chunks = self.split_text_into_chunks(text)
            
            # Генерируем вопросы для каждого блока
            all_qa_pairs = []
            
            # Создаем путь к файлу ошибок
            error_log_path = output_path.replace('.json', '.err')
            
            for i, chunk in enumerate(chunks):
                print(f"\nОбрабатываю блок {i+1}/{len(chunks)}")
                qa_pairs = self.generate_qa_pairs(chunk, questions_per_chunk, error_log_path)
                all_qa_pairs.extend(qa_pairs)
                
                # Небольшая пауза между запросами
                if i < len(chunks) - 1:
                    time.sleep(1)
            
            # Сохраняем результат
            if all_qa_pairs:
                output_data = {
                    "source_pdf": pdf_path,
                    "total_questions": len(all_qa_pairs),
                    "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": self.model,
                    "qa_pairs": all_qa_pairs
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                print(f"\n✅ Успешно создан датасет!")
                print(f"📊 Всего вопросов: {len(all_qa_pairs)}")
                print(f"💾 Сохранено в: {output_path}")
                return True
            else:
                print("❌ Не удалось сгенерировать вопросы")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка при обработке PDF: {e}")
            return False


def main():
    """Основная функция программы"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Генератор датасета вопросов-ответов из PDF с использованием Ollama"
    )
    parser.add_argument("pdf_path", help="Путь к PDF файлу")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата (по умолчанию: output.json)")
    parser.add_argument("-m", "--model", default="llama3.1:8b", help="Модель Ollama (по умолчанию: llama3.1:8b)")
    parser.add_argument("-q", "--questions", type=int, default=3, help="Количество вопросов на текстовый блок (по умолчанию: 3)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL Ollama API (по умолчанию: http://localhost:11434)")
    parser.add_argument("--api-token", help="Токен аутентификации для Ollama API")
    
    args = parser.parse_args()
    
    # Проверяем существование PDF файла
    if not os.path.exists(args.pdf_path):
        print(f"❌ Файл не найден: {args.pdf_path}")
        sys.exit(1)
    
    # Определяем путь для сохранения
    if args.output:
        output_path = args.output
    else:
        pdf_name = Path(args.pdf_path).stem
        output_path = f"{pdf_name}_qa_dataset.json"
    
    print("🚀 Запуск генератора датасета вопросов-ответов")
    print(f"📁 PDF файл: {args.pdf_path}")
    print(f"🤖 Модель: {args.model}")
    print(f"❓ Вопросов на блок: {args.questions}")
    print(f"💾 Выходной файл: {output_path}")
    print("-" * 50)
    
    # Создаем генератор
    generator = PDFToQAGenerator(
        ollama_url=args.ollama_url,
        model=args.model,
        api_token=args.api_token
    )
    
    # Обрабатываем PDF
    success = generator.process_pdf(
        pdf_path=args.pdf_path,
        output_path=output_path,
        questions_per_chunk=args.questions
    )
    
    if success:
        print("\n🎉 Генерация завершена успешно!")
        sys.exit(0)
    else:
        print("\n💥 Произошла ошибка при генерации")
        sys.exit(1)


if __name__ == "__main__":
    main()
