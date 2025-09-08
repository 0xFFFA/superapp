#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Демонстрационный скрипт для быстрого запуска анализа аномалий
"""

from ball_mill_anomaly_analysis import BallMillAnomalyAnalyzer
import matplotlib.pyplot as plt

def quick_demo():
    """Быстрая демонстрация основных возможностей"""
    print("🚀 БЫСТРАЯ ДЕМОНСТРАЦИЯ АНАЛИЗА АНОМАЛИЙ")
    print("=" * 50)
    
    # Создаем анализатор
    analyzer = BallMillAnomalyAnalyzer()
    
    # Генерируем данные за 7 дней
    print("📊 Генерация данных за 7 дней...")
    data = analyzer.generate_test_data(days=7, freq='10min')
    
    # Показываем основные статистики
    print("\n📈 ОСНОВНЫЕ СТАТИСТИКИ:")
    print(f"Мощность: {data['power_kw'].mean():.1f} ± {data['power_kw'].std():.1f} кВт")
    print(f"Вес руды: {data['ore_weight_tons'].mean():.1f} ± {data['ore_weight_tons'].std():.1f} т")
    print(f"Ток классификатора: {data['classifier_current_a'].mean():.1f} ± {data['classifier_current_a'].std():.1f} А")
    
    # Обнаруживаем аномалии
    print("\n🔍 Обнаружение аномалий...")
    iqr_anomalies = analyzer.detect_statistical_anomalies(method='iqr')
    corr_anomalies = analyzer.detect_correlation_anomalies()
    
    # Показываем результаты
    print("\n⚠️ ОБНАРУЖЕННЫЕ АНОМАЛИИ:")
    for param, data in iqr_anomalies.items():
        print(f"{param}: {data['count']} аномалий")
    
    # Простая визуализация
    plt.figure(figsize=(15, 10))
    
    # Временные ряды
    plt.subplot(2, 2, 1)
    plt.plot(data['timestamp'], data['power_kw'], 'b-', alpha=0.7)
    plt.title('Мощность (кВт)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.plot(data['timestamp'], data['ore_weight_tons'], 'g-', alpha=0.7)
    plt.title('Вес руды (т)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.plot(data['timestamp'], data['classifier_current_a'], 'r-', alpha=0.7)
    plt.title('Ток классификатора (А)')
    plt.xticks(rotation=45)
    
    # Эффективность
    plt.subplot(2, 2, 4)
    plt.plot(data['timestamp'], data['power_per_ton'], 'm-', alpha=0.7)
    plt.title('Эффективность (кВт/т)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('quick_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Демонстрация завершена!")
    print("📁 Создан файл: quick_demo.png")

if __name__ == "__main__":
    quick_demo()

