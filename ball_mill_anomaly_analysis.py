#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа аномалий в работе шаровой мельницы
Автор: AI Assistant
Дата: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русского текста
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class BallMillAnomalyAnalyzer:
    """Класс для анализа аномалий в работе шаровой мельницы"""
    
    def __init__(self):
        self.data = None
        self.anomalies = {}
        
    def generate_test_data(self, days=30, freq='5min'):
        """
        Генерирует тестовые данные для шаровой мельницы
        
        Параметры:
        - days: количество дней данных
        - freq: частота измерений
        """
        print("🔄 Генерация тестовых данных для шаровой мельницы...")
        
        # Создаем временной индекс
        start_date = datetime.now() - timedelta(days=days)
        time_index = pd.date_range(start=start_date, periods=days*24*12, freq=freq)  # каждые 5 минут
        
        n_points = len(time_index)
        
        # Базовые значения (нормальная работа)
        base_power = 850  # кВт
        base_ore_weight = 45  # тонн
        base_classifier_current = 120  # А
        
        # Генерируем нормальные данные с небольшими флуктуациями
        power_normal = np.random.normal(base_power, 15, n_points)
        ore_weight_normal = np.random.normal(base_ore_weight, 3, n_points)
        classifier_current_normal = np.random.normal(base_classifier_current, 8, n_points)
        
        # Добавляем циклические компоненты (смена смен, суточные циклы)
        hour_cycle = np.sin(2 * np.pi * np.arange(n_points) / (24*12)) * 10  # суточный цикл
        shift_cycle = np.sin(2 * np.pi * np.arange(n_points) / (8*12)) * 5   # сменный цикл
        
        power_normal += hour_cycle + shift_cycle
        ore_weight_normal += hour_cycle * 0.3
        classifier_current_normal += hour_cycle * 0.5
        
        # Создаем копии для внесения аномалий
        power = power_normal.copy()
        ore_weight = ore_weight_normal.copy()
        classifier_current = classifier_current_normal.copy()
        
        # Вносим различные типы аномалий
        self._add_anomalies(power, ore_weight, classifier_current, n_points)
        
        # Создаем DataFrame
        self.data = pd.DataFrame({
            'timestamp': time_index,
            'power_kw': power,
            'ore_weight_tons': ore_weight,
            'classifier_current_a': classifier_current
        })
        
        # Добавляем производные метрики
        self.data['power_per_ton'] = self.data['power_kw'] / self.data['ore_weight_tons']
        self.data['efficiency_ratio'] = self.data['ore_weight_tons'] / self.data['classifier_current_a']
        
        print(f"✅ Сгенерировано {len(self.data)} записей за {days} дней")
        return self.data
    
    def _add_anomalies(self, power, ore_weight, classifier_current, n_points):
        """Добавляет различные типы аномалий в данные"""
        
        # 1. Внезапные скачки мощности (перегрузка)
        spike_indices = np.random.choice(n_points, size=15, replace=False)
        for idx in spike_indices:
            power[idx:idx+3] += np.random.uniform(100, 200)  # скачок на 3 точки
            
        # 2. Падение веса руды (проблемы с подачей)
        drop_indices = np.random.choice(n_points, size=10, replace=False)
        for idx in drop_indices:
            ore_weight[idx:idx+5] *= 0.3  # резкое падение
            
        # 3. Аномальные значения тока классификатора
        anomaly_indices = np.random.choice(n_points, size=20, replace=False)
        for idx in anomaly_indices:
            if np.random.random() > 0.5:
                classifier_current[idx:idx+2] += np.random.uniform(50, 100)  # скачок
            else:
                classifier_current[idx:idx+2] *= 0.4  # падение
                
        # 4. Долгосрочный тренд деградации (последние 3 дня)
        degradation_start = int(n_points * 0.9)  # последние 10% данных
        degradation_factor = np.linspace(1, 0.7, n_points - degradation_start)
        power[degradation_start:] *= degradation_factor
        classifier_current[degradation_start:] *= (1 + (1 - degradation_factor) * 0.3)
        
        # 5. Циклические аномалии (неисправность оборудования)
        cycle_start = int(n_points * 0.3)
        cycle_end = int(n_points * 0.5)
        cycle_period = 20  # период аномального цикла
        for i in range(cycle_start, cycle_end, cycle_period):
            if i + 10 < cycle_end:
                power[i:i+10] += 80 * np.sin(np.linspace(0, 2*np.pi, 10))
                
        # 6. Корреляционные аномалии (неэффективные соотношения)
        corr_anomaly_indices = np.random.choice(n_points, size=25, replace=False)
        for idx in corr_anomaly_indices:
            # Высокая мощность при низком весе руды
            power[idx] += 150
            ore_weight[idx] *= 0.5
    
    def detect_statistical_anomalies(self, method='iqr', threshold=1.5):
        """
        Обнаружение аномалий статистическими методами
        
        Параметры:
        - method: 'iqr', 'zscore', 'modified_zscore'
        - threshold: пороговое значение
        """
        print(f"🔍 Обнаружение аномалий методом {method}...")
        
        numeric_columns = ['power_kw', 'ore_weight_tons', 'classifier_current_a', 
                          'power_per_ton', 'efficiency_ratio']
        
        anomalies = {}
        
        for column in numeric_columns:
            data = self.data[column]
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomaly_mask = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                anomaly_mask = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = data.median()
                mad = np.median(np.abs(data - median))
                modified_z_scores = 0.6745 * (data - median) / mad
                anomaly_mask = np.abs(modified_z_scores) > threshold
            
            anomalies[column] = {
                'indices': self.data[anomaly_mask].index.tolist(),
                'values': data[anomaly_mask].tolist(),
                'count': anomaly_mask.sum()
            }
        
        self.anomalies[method] = anomalies
        return anomalies
    
    def detect_correlation_anomalies(self, window_size=100):
        """
        Обнаружение корреляционных аномалий
        Анализирует нарушения нормальных соотношений между параметрами
        """
        print("🔗 Обнаружение корреляционных аномалий...")
        
        # Вычисляем скользящие корреляции
        window = window_size
        power_ore_corr = []
        power_current_corr = []
        ore_current_corr = []
        
        for i in range(window, len(self.data)):
            window_data = self.data.iloc[i-window:i]
            
            power_ore_corr.append(window_data['power_kw'].corr(window_data['ore_weight_tons']))
            power_current_corr.append(window_data['power_kw'].corr(window_data['classifier_current_a']))
            ore_current_corr.append(window_data['ore_weight_tons'].corr(window_data['classifier_current_a']))
        
        # Находим аномальные корреляции
        corr_data = pd.DataFrame({
            'power_ore_corr': power_ore_corr,
            'power_current_corr': power_current_corr,
            'ore_current_corr': ore_current_corr
        })
        
        # Аномалии - корреляции, сильно отличающиеся от медианы
        corr_anomalies = {}
        for col in corr_data.columns:
            median_corr = corr_data[col].median()
            mad = np.median(np.abs(corr_data[col] - median_corr))
            threshold = 2.5 * mad
            
            anomaly_mask = np.abs(corr_data[col] - median_corr) > threshold
            corr_anomalies[col] = {
                'indices': corr_data[anomaly_mask].index.tolist(),
                'values': corr_data[col][anomaly_mask].tolist(),
                'count': anomaly_mask.sum()
            }
        
        self.anomalies['correlation'] = corr_anomalies
        return corr_anomalies
    
    def detect_trend_anomalies(self, window_size=200):
        """
        Обнаружение трендовых аномалий
        Ищет неожиданные изменения в трендах параметров
        """
        print("📈 Обнаружение трендовых аномалий...")
        
        numeric_columns = ['power_kw', 'ore_weight_tons', 'classifier_current_a']
        trend_anomalies = {}
        
        for column in numeric_columns:
            data = self.data[column].values
            
            # Вычисляем скользящие тренды
            trends = []
            for i in range(window_size, len(data)):
                window_data = data[i-window_size:i]
                # Простая линейная регрессия для определения тренда
                x = np.arange(len(window_data))
                slope = np.polyfit(x, window_data, 1)[0]
                trends.append(slope)
            
            # Находим аномальные тренды
            trends = np.array(trends)
            median_trend = np.median(trends)
            mad = np.median(np.abs(trends - median_trend))
            threshold = 3 * mad
            
            anomaly_mask = np.abs(trends - median_trend) > threshold
            trend_anomalies[column] = {
                'indices': np.where(anomaly_mask)[0].tolist(),
                'trends': trends[anomaly_mask].tolist(),
                'count': anomaly_mask.sum()
            }
        
        self.anomalies['trend'] = trend_anomalies
        return trend_anomalies
    
    def visualize_anomalies(self, save_plots=True):
        """
        Визуализация данных и обнаруженных аномалий
        """
        print("📊 Создание визуализаций...")
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Основные временные ряды
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(self.data['timestamp'], self.data['power_kw'], 'b-', alpha=0.7, label='Мощность (кВт)')
        plt.title('Мощность шаровой мельницы', fontsize=14, fontweight='bold')
        plt.ylabel('Мощность (кВт)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        plt.plot(self.data['timestamp'], self.data['ore_weight_tons'], 'g-', alpha=0.7, label='Вес руды (т)')
        plt.title('Вес руды', fontsize=14, fontweight='bold')
        plt.ylabel('Вес (тонны)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 2, 3)
        plt.plot(self.data['timestamp'], self.data['classifier_current_a'], 'r-', alpha=0.7, label='Ток классификатора (А)')
        plt.title('Ток классификатора', fontsize=14, fontweight='bold')
        plt.ylabel('Ток (А)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Производные метрики
        ax4 = plt.subplot(3, 2, 4)
        plt.plot(self.data['timestamp'], self.data['power_per_ton'], 'm-', alpha=0.7, label='Мощность/тонна')
        plt.title('Эффективность (кВт/т)', fontsize=14, fontweight='bold')
        plt.ylabel('кВт/т')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Корреляционная матрица
        ax5 = plt.subplot(3, 2, 5)
        corr_matrix = self.data[['power_kw', 'ore_weight_tons', 'classifier_current_a']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
        plt.title('Корреляционная матрица', fontsize=14, fontweight='bold')
        
        # Распределения
        ax6 = plt.subplot(3, 2, 6)
        self.data[['power_kw', 'ore_weight_tons', 'classifier_current_a']].hist(alpha=0.7, ax=ax6)
        plt.title('Распределения параметров', fontsize=14, fontweight='bold')
        plt.legend(['Мощность', 'Вес руды', 'Ток классификатора'])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ball_mill_analysis.png', dpi=300, bbox_inches='tight')
            print("💾 Графики сохранены в ball_mill_analysis.png")
        
        plt.show()
        
        # Детальная визуализация аномалий
        self._plot_anomaly_details()
    
    def _plot_anomaly_details(self):
        """Детальная визуализация обнаруженных аномалий"""
        if not self.anomalies:
            print("⚠️ Аномалии не обнаружены. Сначала запустите методы обнаружения.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        methods = list(self.anomalies.keys())
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, method in enumerate(methods[:4]):
            if method in self.anomalies:
                ax = axes[i]
                
                # Подсчитываем общее количество аномалий по параметрам
                param_counts = {}
                for param, data in self.anomalies[method].items():
                    if isinstance(data, dict) and 'count' in data:
                        param_counts[param] = data['count']
                
                if param_counts:
                    params = list(param_counts.keys())
                    counts = list(param_counts.values())
                    
                    bars = ax.bar(params, counts, color=colors[i], alpha=0.7)
                    ax.set_title(f'Аномалии: {method}', fontweight='bold')
                    ax.set_ylabel('Количество аномалий')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Добавляем значения на столбцы
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('anomaly_details.png', dpi=300, bbox_inches='tight')
        print("💾 Детали аномалий сохранены в anomaly_details.png")
        plt.show()
    
    def generate_report(self):
        """
        Генерация отчета по обнаруженным аномалиям
        """
        print("📋 Генерация отчета...")
        
        report = []
        report.append("=" * 60)
        report.append("ОТЧЕТ ПО АНАЛИЗУ АНОМАЛИЙ ШАРОВОЙ МЕЛЬНИЦЫ")
        report.append("=" * 60)
        report.append(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Период данных: {self.data['timestamp'].min()} - {self.data['timestamp'].max()}")
        report.append(f"Общее количество измерений: {len(self.data)}")
        report.append("")
        
        # Статистика по данным
        report.append("СТАТИСТИКА ПО ПАРАМЕТРАМ:")
        report.append("-" * 40)
        for col in ['power_kw', 'ore_weight_tons', 'classifier_current_a']:
            data = self.data[col]
            report.append(f"{col}:")
            report.append(f"  Среднее: {data.mean():.2f}")
            report.append(f"  Медиана: {data.median():.2f}")
            report.append(f"  Стд. отклонение: {data.std():.2f}")
            report.append(f"  Мин: {data.min():.2f}")
            report.append(f"  Макс: {data.max():.2f}")
            report.append("")
        
        # Анализ аномалий
        if self.anomalies:
            report.append("ОБНАРУЖЕННЫЕ АНОМАЛИИ:")
            report.append("-" * 40)
            
            for method, anomalies in self.anomalies.items():
                report.append(f"\nМетод: {method.upper()}")
                total_anomalies = 0
                
                for param, data in anomalies.items():
                    if isinstance(data, dict) and 'count' in data:
                        count = data['count']
                        total_anomalies += count
                        report.append(f"  {param}: {count} аномалий")
                
                report.append(f"  Всего аномалий: {total_anomalies}")
        
        # Рекомендации
        report.append("\nРЕКОМЕНДАЦИИ:")
        report.append("-" * 40)
        
        if 'iqr' in self.anomalies:
            power_anomalies = self.anomalies['iqr'].get('power_kw', {}).get('count', 0)
            if power_anomalies > 20:
                report.append("⚠️ Высокое количество аномалий мощности - проверить нагрузку мельницы")
            
            ore_anomalies = self.anomalies['iqr'].get('ore_weight_tons', {}).get('count', 0)
            if ore_anomalies > 15:
                report.append("⚠️ Аномалии веса руды - проверить систему подачи")
            
            current_anomalies = self.anomalies['iqr'].get('classifier_current_a', {}).get('count', 0)
            if current_anomalies > 15:
                report.append("⚠️ Аномалии тока классификатора - проверить состояние классификатора")
        
        if 'correlation' in self.anomalies:
            corr_anomalies = sum(data.get('count', 0) for data in self.anomalies['correlation'].values())
            if corr_anomalies > 10:
                report.append("⚠️ Нарушения корреляций - возможны системные проблемы")
        
        report.append("\nОБЩИЕ РЕКОМЕНДАЦИИ:")
        report.append("1. Регулярно контролировать соотношение мощность/вес руды")
        report.append("2. Мониторить тренды деградации оборудования")
        report.append("3. Настроить автоматические алерты при превышении пороговых значений")
        report.append("4. Провести техническое обслуживание при обнаружении трендовых аномалий")
        
        report_text = "\n".join(report)
        
        # Сохраняем отчет
        with open('ball_mill_anomaly_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("💾 Отчет сохранен в ball_mill_anomaly_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """Основная функция для запуска анализа"""
    print("🏭 АНАЛИЗ АНОМАЛИЙ ШАРОВОЙ МЕЛЬНИЦЫ")
    print("=" * 50)
    
    # Создаем анализатор
    analyzer = BallMillAnomalyAnalyzer()
    
    # Генерируем тестовые данные
    data = analyzer.generate_test_data(days=30, freq='5min')
    
    # Обнаруживаем аномалии различными методами
    print("\n🔍 ОБНАРУЖЕНИЕ АНОМАЛИЙ:")
    print("-" * 30)
    
    # Статистические методы
    iqr_anomalies = analyzer.detect_statistical_anomalies(method='iqr', threshold=1.5)
    zscore_anomalies = analyzer.detect_statistical_anomalies(method='zscore', threshold=2.5)
    
    # Корреляционные аномалии
    corr_anomalies = analyzer.detect_correlation_anomalies(window_size=100)
    
    # Трендовые аномалии
    trend_anomalies = analyzer.detect_trend_anomalies(window_size=200)
    
    # Визуализация
    print("\n📊 ВИЗУАЛИЗАЦИЯ:")
    print("-" * 20)
    analyzer.visualize_anomalies(save_plots=True)
    
    # Генерация отчета
    print("\n📋 ОТЧЕТ:")
    print("-" * 15)
    report = analyzer.generate_report()
    
    print("\n✅ Анализ завершен!")
    print("📁 Созданные файлы:")
    print("  - ball_mill_analysis.png (основные графики)")
    print("  - anomaly_details.png (детали аномалий)")
    print("  - ball_mill_anomaly_report.txt (отчет)")

if __name__ == "__main__":
    main()

