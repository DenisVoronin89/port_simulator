"""
============================
 СИМУЛЯЦИОННАЯ МОДЕЛЬ МОРСКОГО ПОРТА
============================

Дискретно-событийная симуляция работы морского порта для анализа
пропускной способности, выявления узких мест и оптимизации логистики.

Компоненты системы:
- PortMetrics: Сбор и агрегация метрик работы порта
- Port: Модель порта с ресурсами (причалы, краны, склады, ЖД)
- ship_process: Процесс обработки судна от прибытия до отшвартовки
- railway_unload_process: Процесс вывоза грузов через ЖД
- ship_generator: Генератор прибытия судов
- run_simulation: Запуск симуляции и получение результатов
- calculate_capacity_metrics: Расчет пропускной способности и узких мест
- main: Streamlit интерфейс для интерактивной работы

Используется библиотека SimPy для дискретно-событийного моделирования.
"""

import simpy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import copy


# Класс для сбора и хранения метрик работы порта
class PortMetrics:

    def __init__(self):
        self.ships_processed = 0  # Счетчик обработанных судов
        self.ships_in_queue = []  # История очередей [{time, queue}]
        self.berth_usage = []  # История загрузки причалов [{time, usage}]
        self.crane_usage = []  # История загрузки кранов [{time, usage}]
        self.storage_usage = []  # История заполненности складов [{time, usage}]
        self.ship_waiting_times = []  # Время ожидания каждого судна (часы)
        self.ship_processing_times = []  # Время обработки каждого судна (часы)
        self.cargo_processed = []  # Количество груза с каждого судна (тонны)
        self.bottlenecks = []  # Список выявленных узких мест

    # Запись прибытия судна и текущей длины очереди.
    def record_ship_arrival(self, time, queue_length):
        self.ships_in_queue.append({'time': time, 'queue': queue_length})

    # Запись завершения обработки судна.
    def record_ship_processing(self, waiting_time, processing_time, cargo_amount):
        self.ships_processed += 1  # Увеличиваем счетчик
        self.ship_waiting_times.append(waiting_time)  # Сохраняем время ожидания
        self.ship_processing_times.append(processing_time)  # Сохраняем время обработки
        self.cargo_processed.append(cargo_amount)  # Сохраняем объем груза

    # Запись текущего использования ресурсов порта.
    def record_resource_usage(self, time, berths_busy, cranes_busy, storage_used, storage_capacity):
        self.berth_usage.append({'time': time, 'usage': berths_busy})
        self.crane_usage.append({'time': time, 'usage': cranes_busy})
        storage_pct = storage_used / storage_capacity * 100 if storage_capacity > 0 else 0  # Процент заполненности
        self.storage_usage.append({'time': time, 'usage': storage_pct})


# Модель морского порта с ресурсами и логикой их использования
class Port:

    def __init__(self, env, config, metrics):
        self.env = env  # Среда симуляции SimPy
        self.config = config  # Конфигурация порта
        self.metrics = metrics  # Сборщик метрик

        # Ресурсы порта
        self.oil_berths = simpy.Resource(env, capacity=config['oil_berths'])  # Нефтяные причалы
        self.dry_berths = simpy.Resource(env, capacity=config['dry_berths'])  # Сухогрузные причалы
        self.cranes = simpy.Resource(env, capacity=config['cranes'])  # Краны для сухих грузов

        # Склады
        self.grain_storage = simpy.Container(env, capacity=config['grain_storage_capacity'], init=0)  # Зерновой терминал
        self.general_storage = simpy.Container(env, capacity=config['general_storage_capacity'], init=0)  # Общий склад
        self.oil_storage = simpy.Container(env, capacity=config['oil_storage_capacity'], init=0)  # Нефтебаза

        # ЖД инфраструктура
        self.railway = simpy.Resource(env, capacity=config['railway_capacity'])  # ЖД пути

        # Вспомогательные счетчики
        self.total_storage_used = 0  # Текущая загрузка всех складов
        self.total_storage_capacity = (config['grain_storage_capacity'] +
                                        config['general_storage_capacity'] +
                                        config['oil_storage_capacity'])  # Общая емкость складов

    # Выбор причала в зависимости от типа груза.
    def get_berth(self, cargo_type):
        if cargo_type == 'oil':
            return self.oil_berths
        else:
            return self.dry_berths

    # Выбор склада в зависимости от типа груза.
    def get_storage(self, cargo_type):
        if cargo_type == 'grain':
            return self.grain_storage
        elif cargo_type == 'oil':
            return self.oil_storage
        else:
            return self.general_storage

    # Получение скорости разгрузки для типа груза (тонн/час).
    def get_unloading_speed(self, cargo_type):
        speeds = {
            'grain': self.config['grain_speed'],  # 300 т/ч (конвейер)
            'oil': self.config['oil_speed'],  # 1000 т/ч (насосы)
            'general': self.config['general_speed']  # 20 т/ч (краны)
        }
        return speeds.get(cargo_type, 20)


# Процесс обработки одного судна в порту.
# Этапы: прибытие → ожидание причала → швартовка → запрос крана → разгрузка → размещение груза → отшвартовка.
def ship_process(env, ship_id, port, cargo_type, cargo_amount, metrics):
    arrival_time = env.now  # Время прибытия судна

    berth = port.get_berth(cargo_type)
    queue_length = len(berth.queue)
    metrics.record_ship_arrival(env.now, queue_length)

    # Ожидание и швартовка к причалу
    with berth.request() as berth_req:
        yield berth_req
        berth_wait_time = env.now - arrival_time

        # Разгрузка (логика зависит от типа груза)
        if cargo_type != 'oil':
            # Для сухих грузов нужен кран
            with port.cranes.request() as crane_req:
                yield crane_req

                storage = port.get_storage(cargo_type)

                # Ожидание освобождения места на складе
                while storage.level + cargo_amount > storage.capacity:
                    yield env.timeout(1)  # Ждем 1 час

                # Разгрузка судна
                unloading_speed = port.get_unloading_speed(cargo_type)
                unloading_time = cargo_amount / unloading_speed
                yield env.timeout(unloading_time)

                # Размещение на складе
                yield storage.put(cargo_amount)
        else:
            # Для нефти - прямая перекачка без крана
            storage = port.get_storage(cargo_type)

            while storage.level + cargo_amount > storage.capacity:
                yield env.timeout(1)

            unloading_speed = port.get_unloading_speed(cargo_type)
            unloading_time = cargo_amount / unloading_speed
            yield env.timeout(unloading_time)

            yield storage.put(cargo_amount)

    # Запись завершения обработки
    processing_time = env.now - arrival_time
    metrics.record_ship_processing(berth_wait_time, processing_time, cargo_amount)

    # Периодический сбор метрик использования ресурсов
    if ship_id % 10 == 0:
        berths_busy = (port.oil_berths.count + port.dry_berths.count)
        cranes_busy = port.cranes.count
        storage_used = (port.grain_storage.level + port.general_storage.level + port.oil_storage.level)
        metrics.record_resource_usage(env.now, berths_busy, cranes_busy, storage_used, port.total_storage_capacity)


# Процесс вывоза грузов со складов через ЖД.
# Приоритет вывоза: Зерно > Генгрузы > Нефть.
def railway_unload_process(env, port, metrics):
    while True:
        interval = 24 / port.config['trains_per_day']  # Интервал между поездами (часы)
        yield env.timeout(interval)

        with port.railway.request() as rail_req:
            yield rail_req

            unload_amount = port.config['train_capacity']  # Вместимость поезда

            # Вывоз с приоритетом
            if port.grain_storage.level >= unload_amount:
                yield port.grain_storage.get(unload_amount)
            elif port.general_storage.level >= unload_amount:
                yield port.general_storage.get(unload_amount)
            elif port.oil_storage.level >= unload_amount:
                yield port.oil_storage.get(unload_amount)
            else:
                # Берем что есть со всех складов
                amount = min(unload_amount, port.grain_storage.level + port.general_storage.level + port.oil_storage.level)
                if amount > 0:
                    if port.grain_storage.level > 0:
                        take = min(amount, port.grain_storage.level)
                        yield port.grain_storage.get(take)
                        amount -= take
                    if amount > 0 and port.general_storage.level > 0:
                        take = min(amount, port.general_storage.level)
                        yield port.general_storage.get(take)
                        amount -= take
                    if amount > 0 and port.oil_storage.level > 0:
                        take = min(amount, port.oil_storage.level)
                        yield port.oil_storage.get(take)

            yield env.timeout(2)  # Время загрузки поезда


# Генератор прибытия судов в порт.
# Использует экспоненциальное распределение для моделирования пуассоновского потока.
def ship_generator(env, port, metrics, ships_per_year, cargo_distribution):
    ship_id = 0
    inter_arrival_time = 8760 / ships_per_year  # Средний интервал между судами (часы)

    while True:
        # Определяем тип груза согласно распределению
        cargo_type = np.random.choice(
            ['grain', 'oil', 'general'],
            p=[cargo_distribution['grain'], cargo_distribution['oil'], cargo_distribution['general']]
        )

        # Определяем количество груза
        if cargo_type == 'grain':
            cargo_amount = np.random.uniform(3000, 8000)  # Средний зерновоз
        elif cargo_type == 'oil':
            cargo_amount = np.random.uniform(5000, 15000)  # Средний танкер
        else:
            cargo_amount = np.random.uniform(1000, 5000)  # Контейнеровоз/балкер

        env.process(ship_process(env, ship_id, port, cargo_type, cargo_amount, metrics))

        ship_id += 1

        # Интервал до следующего судна
        yield env.timeout(np.random.exponential(inter_arrival_time))


# Запуск дискретно-событийной симуляции работы порта.
def run_simulation(config, simulation_hours=8760):
    env = simpy.Environment()  # Создаем среду симуляции
    metrics = PortMetrics()  # Инициализируем сборщик метрик
    port = Port(env, config, metrics)  # Создаем модель порта

    # Запуск процессов
    env.process(ship_generator(env, port, metrics, config['ships_per_year'], config['cargo_distribution']))
    env.process(railway_unload_process(env, port, metrics))

    # Выполнение симуляции
    env.run(until=simulation_hours)

    return metrics, port


# Расчет метрик пропускной способности и выявление узких мест.
def calculate_capacity_metrics(metrics, port, config, simulation_hours):
    # Обработанный груз
    total_cargo = sum(metrics.cargo_processed)
    annual_cargo = total_cargo * (8760 / simulation_hours)  # Экстраполяция на год

    # Время ожидания и обработки
    avg_wait_time = np.mean(metrics.ship_waiting_times) if metrics.ship_waiting_times else 0
    avg_processing_time = np.mean(metrics.ship_processing_times) if metrics.ship_processing_times else 0

    # Загрузка причалов
    avg_berth_usage = np.mean([x['usage'] for x in metrics.berth_usage]) if metrics.berth_usage else 0
    total_berths = config['oil_berths'] + config['dry_berths']
    berth_utilization = (avg_berth_usage / total_berths * 100) if total_berths > 0 else 0

    # Загрузка кранов
    avg_crane_usage = np.mean([x['usage'] for x in metrics.crane_usage]) if metrics.crane_usage else 0
    crane_utilization = (avg_crane_usage / config['cranes'] * 100) if config['cranes'] > 0 else 0

    # Загрузка складов
    avg_storage_usage = np.mean([x['usage'] for x in metrics.storage_usage]) if metrics.storage_usage else 0

    # Очереди
    avg_queue = np.mean([x['queue'] for x in metrics.ships_in_queue]) if metrics.ships_in_queue else 0
    max_queue = max([x['queue'] for x in metrics.ships_in_queue]) if metrics.ships_in_queue else 0

    # Теоретическая пропускная способность элементов
    avg_ship_time = avg_processing_time if avg_processing_time > 0 else 24
    ships_per_year_capacity = total_berths * 8760 / avg_ship_time
    avg_cargo_per_ship = total_cargo / metrics.ships_processed if metrics.ships_processed > 0 else 5000
    berth_capacity_annual = ships_per_year_capacity * avg_cargo_per_ship  # Пропускная способность причалов

    crane_capacity_annual = config['cranes'] * config['general_speed'] * 8760 * 0.7  # Пропускная способность кранов (70% эффективность)

    total_storage_capacity = (config['grain_storage_capacity'] +
                               config['general_storage_capacity'] +
                               config['oil_storage_capacity'])
    storage_capacity_annual = total_storage_capacity * 24  # Пропускная способность складов (24 оборота/год)

    railway_capacity_annual = config['trains_per_day'] * config['train_capacity'] * 365  # Пропускная способность ЖД

    # Определение узкого места (бутылочного горлышка)
    capacities = {
        'Причалы': berth_capacity_annual,
        'Краны': crane_capacity_annual,
        'Склады': storage_capacity_annual,
        'ЖД': railway_capacity_annual
    }

    theoretical_max_capacity = min(capacities.values())  # Минимальный элемент - ограничитель системы
    bottleneck = min(capacities, key=capacities.get)  # Узкое место

    # Коэффициент использования и запас прочности
    capacity_utilization = (annual_cargo / theoretical_max_capacity * 100) if theoretical_max_capacity > 0 else 0
    reserve_capacity = max(0, theoretical_max_capacity - annual_cargo)

    # Анализ признаков отказа
    critical_issues = []

    # Проверка критических показателей
    if max_queue > 5:
        critical_issues.append(f"Критическая очередь судов: {int(max_queue)} (норма < 5)")

    if avg_wait_time > 72:
        critical_issues.append(f"Критический простой судов: {avg_wait_time:.1f}ч (норма < 72ч)")

    if avg_storage_usage > 95:
        critical_issues.append(f"Критическое заполнение складов: {avg_storage_usage:.1f}% (норма < 95%)")

    # Определение состояния системы
    if capacity_utilization > 80:
        system_status = "На пределе"
        status_color = "red"
    elif capacity_utilization > 60:
        system_status = "Умеренная нагрузка"
        status_color = "orange"
    else:
        system_status = "Есть значительный резерв"
        status_color = "green"

    return {
        'annual_cargo_tons': annual_cargo,
        'ships_processed': metrics.ships_processed,
        'avg_wait_time_hours': avg_wait_time,
        'avg_processing_time_hours': avg_processing_time,
        'berth_utilization_pct': berth_utilization,
        'crane_utilization_pct': crane_utilization,
        'storage_utilization_pct': avg_storage_usage,
        'avg_queue_length': avg_queue,
        'max_queue_length': max_queue,
        'theoretical_max_capacity_tons': theoretical_max_capacity,
        'capacity_utilization_pct': capacity_utilization,
        'reserve_capacity_tons': reserve_capacity,
        'bottleneck': bottleneck,
        'capacities': capacities,
        'critical_issues': critical_issues,
        'system_status': system_status,
        'status_color': status_color
    }


# Анализ точки коллапса через постепенное увеличение нагрузки
def find_collapse_point(config, simulation_hours=8760):
    """
    Постепенно увеличивает нагрузку до появления признаков коллапса.
    Возвращает пороговые значения нагрузки.
    """
    results = []
    load_multipliers = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]

    first_issues_load = None
    constant_delays_load = None
    full_collapse_load = None

    base_ships = config['ships_per_year']

    for multiplier in load_multipliers:
        # Увеличиваем нагрузку
        test_config = copy.deepcopy(config)
        test_config['ships_per_year'] = int(base_ships * multiplier)

        # Запускаем симуляцию
        metrics, port = run_simulation(test_config, simulation_hours)
        capacity_metrics = calculate_capacity_metrics(metrics, port, test_config, simulation_hours)

        # Проверяем признаки проблем
        has_critical_queue = capacity_metrics['max_queue_length'] > 5
        has_critical_wait = capacity_metrics['avg_wait_time_hours'] > 72
        has_critical_storage = capacity_metrics['storage_utilization_pct'] > 95

        # Считаем количество критических показателей
        critical_count = sum([has_critical_queue, has_critical_wait, has_critical_storage])

        # Определяем пороги
        if first_issues_load is None and critical_count >= 1:
            first_issues_load = multiplier

        if constant_delays_load is None and critical_count >= 2:
            constant_delays_load = multiplier

        if full_collapse_load is None and critical_count >= 3:
            full_collapse_load = multiplier

        results.append({
            'load_multiplier': multiplier,
            'annual_cargo_tons': capacity_metrics['annual_cargo_tons'],
            'avg_queue': capacity_metrics['avg_queue_length'],
            'max_queue': capacity_metrics['max_queue_length'],
            'avg_wait_time': capacity_metrics['avg_wait_time_hours'],
            'storage_usage': capacity_metrics['storage_utilization_pct'],
            'critical_count': critical_count,
            'critical_issues': capacity_metrics['critical_issues']
        })

    return {
        'results': results,
        'first_issues_load': first_issues_load,
        'constant_delays_load': constant_delays_load,
        'full_collapse_load': full_collapse_load
    }


# Запуск сценариев нагрузочного тестирования
def run_stress_scenarios(base_config, simulation_hours=4380):
    """
    Запускает 6 сценариев из ТЗ для оценки устойчивости системы.
    """
    scenarios = {}

    # Сценарий 1: Постепенное увеличение нагрузки на 10% ежемесячно
    # (упрощенно: 110% от текущей)
    scenario1_config = copy.deepcopy(base_config)
    scenario1_config['ships_per_year'] = int(base_config['ships_per_year'] * 1.1)
    metrics1, port1 = run_simulation(scenario1_config, simulation_hours)
    scenarios['scenario_1'] = {
        'name': 'Увеличение нагрузки на 10%',
        'config': scenario1_config,
        'metrics': calculate_capacity_metrics(metrics1, port1, scenario1_config, simulation_hours)
    }

    # Сценарий 2: Пиковая нагрузка (зерно + нефть одновременно)
    scenario2_config = copy.deepcopy(base_config)
    scenario2_config['cargo_distribution'] = {'grain': 0.5, 'oil': 0.45, 'general': 0.05}
    scenario2_config['ships_per_year'] = int(base_config['ships_per_year'] * 1.3)
    metrics2, port2 = run_simulation(scenario2_config, simulation_hours)
    scenarios['scenario_2'] = {
        'name': 'Пиковая нагрузка (зерно+нефть)',
        'config': scenario2_config,
        'metrics': calculate_capacity_metrics(metrics2, port2, scenario2_config, simulation_hours)
    }

    # Сценарий 3: Аварийная ситуация (выход из строя 2 кранов)
    scenario3_config = copy.deepcopy(base_config)
    scenario3_config['cranes'] = max(1, base_config['cranes'] - 2)
    metrics3, port3 = run_simulation(scenario3_config, simulation_hours)
    scenarios['scenario_3'] = {
        'name': 'Авария: выход из строя 2 кранов',
        'config': scenario3_config,
        'metrics': calculate_capacity_metrics(metrics3, port3, scenario3_config, simulation_hours)
    }

    # Сценарий 4: Работа при 50% доступности оборудования
    scenario4_config = copy.deepcopy(base_config)
    scenario4_config['cranes'] = max(1, base_config['cranes'] // 2)
    scenario4_config['oil_berths'] = max(1, base_config['oil_berths'] // 2)
    scenario4_config['dry_berths'] = max(1, base_config['dry_berths'] // 2)
    metrics4, port4 = run_simulation(scenario4_config, simulation_hours)
    scenarios['scenario_4'] = {
        'name': 'Работа при 50% доступности',
        'config': scenario4_config,
        'metrics': calculate_capacity_metrics(metrics4, port4, scenario4_config, simulation_hours)
    }

    # Сценарий 5: Увеличение времени обработки на 30% (плохая погода)
    scenario5_config = copy.deepcopy(base_config)
    scenario5_config['grain_speed'] = int(base_config['grain_speed'] * 0.7)
    scenario5_config['oil_speed'] = int(base_config['oil_speed'] * 0.7)
    scenario5_config['general_speed'] = int(base_config['general_speed'] * 0.7)
    metrics5, port5 = run_simulation(scenario5_config, simulation_hours)
    scenarios['scenario_5'] = {
        'name': 'Снижение скорости на 30% (погода)',
        'config': scenario5_config,
        'metrics': calculate_capacity_metrics(metrics5, port5, scenario5_config, simulation_hours)
    }

    # Сценарий 6: Одновременный приход 3 крупных судов
    # (моделируется через резкое увеличение частоты судов)
    scenario6_config = copy.deepcopy(base_config)
    scenario6_config['ships_per_year'] = int(base_config['ships_per_year'] * 1.5)
    metrics6, port6 = run_simulation(scenario6_config, simulation_hours)
    scenarios['scenario_6'] = {
        'name': 'Одновременный приход крупных судов',
        'config': scenario6_config,
        'metrics': calculate_capacity_metrics(metrics6, port6, scenario6_config, simulation_hours)
    }

    return scenarios


# Функция для ответов на конкретные вопросы из ТЗ
def answer_key_questions(results, config):
    """
    Отвечает на конкретные вопросы из технического задания.
    """
    answers = {}

    # 1. Какова фактическая пропускная способность порта сегодня?
    answers['current_capacity'] = {
        'max_tons_per_year': results['theoretical_max_capacity_tons'],
        'optimal_load': results['theoretical_max_capacity_tons'] * 0.75,  # 75% для оптимальной работы
        'current_load': results['annual_cargo_tons']
    }

    # 2. Какой элемент инфраструктуры является главным ограничителем?
    capacities_sorted = sorted(results['capacities'].items(), key=lambda x: x[1])
    answers['bottleneck_analysis'] = {
        'main_bottleneck': results['bottleneck'],
        'ranking': [{'element': k, 'capacity': v} for k, v in capacities_sorted],
        'effect_of_removal': {}
    }

    # Оценка эффекта от устранения каждого ограничения
    for element, capacity in results['capacities'].items():
        remaining_capacities = [v for k, v in results['capacities'].items() if k != element]
        if remaining_capacities:
            new_capacity = min(remaining_capacities)
            improvement = new_capacity - results['theoretical_max_capacity_tons']
            answers['bottleneck_analysis']['effect_of_removal'][element] = {
                'new_capacity': new_capacity,
                'improvement': improvement,
                'improvement_pct': (improvement / results['theoretical_max_capacity_tons'] * 100) if results['theoretical_max_capacity_tons'] > 0 else 0
            }

    # 3. Выдержит ли порт плановые 18 млн тонн?
    target_capacity = 18_000_000
    can_handle = results['theoretical_max_capacity_tons'] >= target_capacity
    deficit = max(0, target_capacity - results['theoretical_max_capacity_tons'])

    answers['target_18m'] = {
        'can_handle': can_handle,
        'deficit': deficit,
        'deficit_pct': (deficit / target_capacity * 100) if not can_handle else 0,
        'required_improvements': []
    }

    # Определяем необходимые изменения
    if not can_handle:
        for element, capacity in capacities_sorted:
            if capacity < target_capacity:
                required_increase = target_capacity - capacity
                required_increase_pct = (required_increase / capacity * 100)
                answers['target_18m']['required_improvements'].append({
                    'element': element,
                    'current': capacity,
                    'required': target_capacity,
                    'increase_needed': required_increase,
                    'increase_pct': required_increase_pct
                })

    # 4. Каков запас прочности системы?
    answers['reserve_strength'] = {
        'reserve_tons': results['reserve_capacity_tons'],
        'current_utilization_pct': results['capacity_utilization_pct'],
        'max_increase_without_investment': results['reserve_capacity_tons'],
        'max_increase_pct': ((results['reserve_capacity_tons'] / results['annual_cargo_tons']) * 100) if results['annual_cargo_tons'] > 0 else 0,
        'critical_growth_pct': None
    }

    # Критический процент роста
    if results['capacity_utilization_pct'] < 80:
        critical_growth = ((80 - results['capacity_utilization_pct']) / results['capacity_utilization_pct'] * 100) if results['capacity_utilization_pct'] > 0 else 0
        answers['reserve_strength']['critical_growth_pct'] = critical_growth
    else:
        answers['reserve_strength']['critical_growth_pct'] = 0

    return answers


# Главная функция Streamlit приложения для интерактивной работы с моделью.
def main():
    st.set_page_config(page_title="Симуляция Порта", layout="wide")

    st.title("🚢 Симуляционная модель морского порта Лена")
    st.markdown("---")

    # Параметры конфигурации
    st.sidebar.header("⚙️ Параметры порта")

    # Причалы
    st.sidebar.subheader("Причалы")
    oil_berths = st.sidebar.number_input("Нефтяные причалы", min_value=1, max_value=10, value=5)
    dry_berths = st.sidebar.number_input("Сухогрузные причалы", min_value=1, max_value=10, value=5)

    # Перегрузочное оборудование
    st.sidebar.subheader("Перегрузочное оборудование")
    cranes = st.sidebar.number_input("Краны", min_value=1, max_value=20, value=7)
    grain_speed = st.sidebar.number_input("Скорость разгрузки зерна (т/ч)", min_value=100, max_value=500, value=300)
    oil_speed = st.sidebar.number_input("Скорость разгрузки нефти (т/ч)", min_value=500, max_value=2000, value=1000)
    general_speed = st.sidebar.number_input("Скорость разгрузки генгрузов (т/ч)", min_value=10, max_value=50, value=20)

    # Склады
    st.sidebar.subheader("Склады (тонн)")
    grain_storage = st.sidebar.number_input("Емкость зернового склада", min_value=10000, max_value=200000, value=100000, step=10000)
    general_storage = st.sidebar.number_input("Емкость общего склада", min_value=5000, max_value=100000, value=20000, step=5000)
    oil_storage = st.sidebar.number_input("Емкость нефтебазы", min_value=100000, max_value=1000000, value=540000, step=50000)

    # ЖД инфраструктура
    st.sidebar.subheader("ЖД инфраструктура")
    trains_per_day = st.sidebar.number_input("Поездов в сутки", min_value=1, max_value=20, value=7)
    train_capacity = st.sidebar.number_input("Вместимость поезда (тонн)", min_value=500, max_value=5000, value=2000, step=500)
    railway_capacity = st.sidebar.number_input("Количество ЖД путей", min_value=1, max_value=10, value=2)

    # Нагрузка
    st.sidebar.subheader("Нагрузка")
    target_cargo = st.sidebar.number_input("Целевой грузооборот (млн тонн/год)", min_value=1.0, max_value=25.0, value=3.4, step=0.5)
    ships_per_year = st.sidebar.number_input("Судозаходов в год", min_value=100, max_value=3000, value=714, step=50)

    # Распределение грузов
    st.sidebar.subheader("Распределение грузов (%)")
    grain_pct = st.sidebar.slider("Зерно", 0, 100, 32)
    oil_pct = st.sidebar.slider("Нефть", 0, 100, 35)
    general_pct = 100 - grain_pct - oil_pct
    st.sidebar.write(f"Генгрузы: {general_pct}%")

    # Параметры симуляции
    st.sidebar.subheader("Симуляция")
    simulation_months = st.sidebar.slider("Период симуляции (месяцев)", 1, 12, 12)
    simulation_hours = int(simulation_months * 730)

    # Формирование конфигурации
    config = {
        'oil_berths': oil_berths,
        'dry_berths': dry_berths,
        'cranes': cranes,
        'grain_speed': grain_speed,
        'oil_speed': oil_speed,
        'general_speed': general_speed,
        'grain_storage_capacity': grain_storage,
        'general_storage_capacity': general_storage,
        'oil_storage_capacity': oil_storage,
        'trains_per_day': trains_per_day,
        'train_capacity': train_capacity,
        'railway_capacity': railway_capacity,
        'ships_per_year': ships_per_year,
        'cargo_distribution': {
            'grain': grain_pct / 100,
            'oil': oil_pct / 100,
            'general': general_pct / 100
        }
    }

    # Запуск симуляции
    run_simulation_btn = st.sidebar.button("🚀 Запустить симуляцию", type="primary")
    if run_simulation_btn:
        with st.spinner("Выполняется симуляция..."):
            metrics, port = run_simulation(config, simulation_hours)
            results = calculate_capacity_metrics(metrics, port, config, simulation_hours)

            st.session_state['results'] = results
            st.session_state['metrics'] = metrics
            st.session_state['config'] = config
            st.rerun()

    # Анализ точки коллапса
    run_collapse_btn = st.sidebar.button("🔥 Найти точку коллапса", type="secondary")
    if run_collapse_btn:
        with st.spinner("Анализ точки коллапса... Это может занять несколько минут"):
            collapse_data = find_collapse_point(config, simulation_hours=2190)
            st.session_state['collapse_data'] = collapse_data
            st.rerun()

    # Запуск стресс-тестирования
    run_stress_btn = st.sidebar.button("⚡ Стресс-тестирование", type="secondary")
    if run_stress_btn:
        with st.spinner("Выполняется стресс-тестирование... Запуск 6 сценариев"):
            stress_scenarios = run_stress_scenarios(config, simulation_hours=2190)
            st.session_state['stress_scenarios'] = stress_scenarios
            st.rerun()

    # Кнопка очистки всех результатов
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Очистить все результаты"):
        if 'results' in st.session_state:
            del st.session_state['results']
        if 'metrics' in st.session_state:
            del st.session_state['metrics']
        if 'collapse_data' in st.session_state:
            del st.session_state['collapse_data']
        if 'stress_scenarios' in st.session_state:
            del st.session_state['stress_scenarios']
        st.rerun()

    # Отображение результатов
    if 'results' in st.session_state:
        results = st.session_state['results']
        metrics = st.session_state['metrics']
        config = st.session_state['config']

        st.success("✅ Симуляция завершена!")

        # Основные метрики
        st.header("📊 Основные показатели")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Грузооборот",
                f"{results['annual_cargo_tons']/1_000_000:.2f} млн т/год",
                f"{((results['annual_cargo_tons']/1_000_000) - target_cargo):.2f} млн т"
            )

        with col2:
            st.metric(
                "Обработано судов",
                f"{results['ships_processed']}",
                f"{results['ships_processed'] - ships_per_year}"
            )

        with col3:
            color = "🟢" if results['avg_wait_time_hours'] < 24 else "🟡" if results['avg_wait_time_hours'] < 72 else "🔴"
            st.metric(
                f"{color} Среднее ожидание",
                f"{results['avg_wait_time_hours']:.1f} ч"
            )

        with col4:
            color = "🟢" if results['max_queue_length'] < 5 else "🟡" if results['max_queue_length'] < 10 else "🔴"
            st.metric(
                f"{color} Макс. очередь",
                f"{int(results['max_queue_length'])} судов"
            )

        st.markdown("---")

        # Критические проблемы (если есть)
        if results.get('critical_issues'):
            st.header("🚨 Критические проблемы")
            for issue in results['critical_issues']:
                st.error(f"⚠️ {issue}")
            st.markdown("---")

        # Пропускная способность
        st.header("⚡ Пропускная способность")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Текущая vs. Теоретическая")

            capacity_data = pd.DataFrame({
                'Тип': ['Текущая', 'Максимальная', 'Цель (18 млн т)'],
                'Значение': [
                    results['annual_cargo_tons'] / 1_000_000,
                    results['theoretical_max_capacity_tons'] / 1_000_000,
                    18
                ]
            })

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=capacity_data['Тип'],
                y=capacity_data['Значение'],
                marker_color=['blue', 'green', 'red'],
                text=capacity_data['Значение'].round(2),
                textposition='outside'
            ))
            fig.update_layout(yaxis_title="Млн тонн/год", height=400)
            st.plotly_chart(fig, use_container_width=True)

            utilization = results['capacity_utilization_pct']
            if utilization < 60:
                st.success(f"✅ Система имеет значительный резерв ({100-utilization:.1f}%)")
            elif utilization < 80:
                st.warning(f"⚠️ Система работает с умеренной нагрузкой ({utilization:.1f}%)")
            else:
                st.error(f"🔴 Система работает на пределе ({utilization:.1f}%)")

        with col2:
            st.subheader("Узкие места системы")

            capacities_df = pd.DataFrame({
                'Элемент': list(results['capacities'].keys()),
                'Пропускная способность (млн т/год)': [v/1_000_000 for v in results['capacities'].values()]
            })
            capacities_df = capacities_df.sort_values('Пропускная способность (млн т/год)')

            fig = go.Figure()
            colors = ['red' if x == results['bottleneck'] else 'lightblue' for x in capacities_df['Элемент']]
            fig.add_trace(go.Bar(
                y=capacities_df['Элемент'],
                x=capacities_df['Пропускная способность (млн т/год)'],
                orientation='h',
                marker_color=colors,
                text=capacities_df['Пропускная способность (млн т/год)'].round(2),
                textposition='outside'
            ))
            fig.update_layout(xaxis_title="Млн тонн/год", height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.error(f"🔴 Главное узкое место: **{results['bottleneck']}**")
            st.info(f"💡 Резерв пропускной способности: **{results['reserve_capacity_tons']/1_000_000:.2f} млн т/год**")

        st.markdown("---")

        # Загрузка ресурсов
        st.header("📈 Загрузка ресурсов")

        col1, col2, col3 = st.columns(3)

        with col1:
            utilization = results['berth_utilization_pct']
            color = "🟢" if utilization < 70 else "🟡" if utilization < 85 else "🔴"
            st.metric(f"{color} Причалы", f"{utilization:.1f}%")

        with col2:
            utilization = results['crane_utilization_pct']
            color = "🟢" if utilization < 70 else "🟡" if utilization < 85 else "🔴"
            st.metric(f"{color} Краны", f"{utilization:.1f}%")

        with col3:
            utilization = results['storage_utilization_pct']
            color = "🟢" if utilization < 70 else "🟡" if utilization < 85 else "🔴"
            st.metric(f"{color} Склады", f"{utilization:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            if metrics.berth_usage:
                berth_df = pd.DataFrame(metrics.berth_usage)
                fig = px.line(berth_df, x='time', y='usage', title='Загрузка причалов во времени')
                fig.update_xaxes(title='Часы симуляции')
                fig.update_yaxes(title='Занято причалов')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if metrics.storage_usage:
                storage_df = pd.DataFrame(metrics.storage_usage)
                fig = px.line(storage_df, x='time', y='usage', title='Заполненность складов во времени')
                fig.update_xaxes(title='Часы симуляции')
                fig.update_yaxes(title='Заполненность (%)')
                fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Критический уровень 80%")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Очереди и время обработки
        st.header("⏱️ Очереди и время обработки")

        col1, col2 = st.columns(2)

        with col1:
            if metrics.ships_in_queue:
                queue_df = pd.DataFrame(metrics.ships_in_queue)
                fig = px.line(queue_df, x='time', y='queue', title='Длина очереди судов')
                fig.update_xaxes(title='Часы симуляции')
                fig.update_yaxes(title='Судов в очереди')
                fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Критический уровень")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if metrics.ship_waiting_times:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=metrics.ship_waiting_times, nbinsx=30, name='Время ожидания'))
                fig.update_layout(
                    title='Распределение времени ожидания судов',
                    xaxis_title='Часы',
                    yaxis_title='Количество судов'
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Выводы и рекомендации
        st.header("💡 Выводы и рекомендации")

        # Получаем ответы на ключевые вопросы
        key_answers = answer_key_questions(results, config)

        # Раздел с ответами на ключевые вопросы
        st.subheader("📋 Ответы на ключевые вопросы")

        # Вопрос 1: Какова фактическая пропускная способность порта?
        with st.expander("1️⃣ Какова фактическая пропускная способность порта сегодня?", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Максимум без сбоев",
                    f"{key_answers['current_capacity']['max_tons_per_year']/1_000_000:.2f} млн т/год"
                )
            with col2:
                st.metric(
                    "Оптимальная нагрузка",
                    f"{key_answers['current_capacity']['optimal_load']/1_000_000:.2f} млн т/год",
                    help="75% от максимальной мощности для эффективной работы"
                )
            with col3:
                st.metric(
                    "Текущая нагрузка",
                    f"{key_answers['current_capacity']['current_load']/1_000_000:.2f} млн т/год"
                )

        # Вопрос 2: Какой элемент является главным ограничителем?
        with st.expander("2️⃣ Какой элемент инфраструктуры является главным ограничителем?", expanded=True):
            st.error(f"**Главное узкое место:** {key_answers['bottleneck_analysis']['main_bottleneck']}")

            st.write("**Рейтинг узких мест (от самого узкого):**")
            ranking_data = []
            for item in key_answers['bottleneck_analysis']['ranking']:
                ranking_data.append({
                    'Элемент': item['element'],
                    'Пропускная способность (млн т/год)': f"{item['capacity']/1_000_000:.2f}"
                })
            st.table(pd.DataFrame(ranking_data))

            st.write("**Эффект от устранения каждого ограничения:**")
            effect_data = []
            for element, data in key_answers['bottleneck_analysis']['effect_of_removal'].items():
                effect_data.append({
                    'Элемент': element,
                    'Новая мощность (млн т)': f"{data['new_capacity']/1_000_000:.2f}",
                    'Прирост (млн т)': f"{data['improvement']/1_000_000:.2f}",
                    'Прирост (%)': f"{data['improvement_pct']:.1f}%"
                })
            st.table(pd.DataFrame(effect_data))

        # Вопрос 3: Выдержит ли порт 18 млн тонн?
        with st.expander("3️⃣ Выдержит ли порт плановые 18 млн тонн?", expanded=True):
            if key_answers['target_18m']['can_handle']:
                st.success("✅ **Порт МОЖЕТ обработать 18 млн тонн/год** при текущей конфигурации")
            else:
                st.error("❌ **Порт НЕ МОЖЕТ обработать 18 млн тонн/год**")
                st.error(f"**Дефицит:** {key_answers['target_18m']['deficit']/1_000_000:.2f} млн т/год ({key_answers['target_18m']['deficit_pct']:.1f}%)")

                if key_answers['target_18m']['required_improvements']:
                    st.write("**Необходимые изменения:**")
                    improvements_data = []
                    for imp in key_answers['target_18m']['required_improvements']:
                        improvements_data.append({
                            'Элемент': imp['element'],
                            'Текущая мощность (млн т)': f"{imp['current']/1_000_000:.2f}",
                            'Требуется (млн т)': f"{imp['required']/1_000_000:.2f}",
                            'Увеличение (млн т)': f"{imp['increase_needed']/1_000_000:.2f}",
                            'Увеличение (%)': f"{imp['increase_pct']:.1f}%"
                        })
                    st.table(pd.DataFrame(improvements_data))

        # Вопрос 4: Каков запас прочности?
        with st.expander("4️⃣ Каков запас прочности системы?", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Резерв без инвестиций",
                    f"{key_answers['reserve_strength']['reserve_tons']/1_000_000:.2f} млн т/год",
                    f"+{key_answers['reserve_strength']['max_increase_pct']:.1f}%"
                )
            with col2:
                if key_answers['reserve_strength']['critical_growth_pct'] and key_answers['reserve_strength']['critical_growth_pct'] > 0:
                    st.metric(
                        "Критический % роста",
                        f"+{key_answers['reserve_strength']['critical_growth_pct']:.1f}%",
                        help="Рост нагрузки, при котором система перейдёт в критическую зону (>80%)"
                    )
                else:
                    st.error("Система уже в критической зоне!")

            st.info(f"**Текущая загрузка системы:** {key_answers['reserve_strength']['current_utilization_pct']:.1f}%")

        st.markdown("---")

        can_handle_18 = results['theoretical_max_capacity_tons'] >= 18_000_000

        if can_handle_18:
            st.success("✅ **Порт может обработать 18 млн тонн/год** при текущей конфигурации")
        else:
            deficit = (18_000_000 - results['theoretical_max_capacity_tons']) / 1_000_000
            st.error(f"❌ **Порт НЕ может обработать 18 млн тонн/год**")
            st.error(f"Дефицит пропускной способности: **{deficit:.2f} млн т/год**")

        st.subheader("Рекомендации по устранению узких мест:")

        bottleneck = results['bottleneck']

        if bottleneck == 'Причалы':
            st.warning("🔴 **Главное узкое место: Причалы**")
            st.write("Рекомендации:")
            st.write("- Увеличить количество причалов на 2-3 единицы")
            st.write("- Оптимизировать расписание судозаходов")
            st.write("- Ускорить процесс швартовки")

        elif bottleneck == 'Краны':
            st.warning("🔴 **Главное узкое место: Краны**")
            st.write("Рекомендации:")
            st.write("- Добавить 3-5 кранов")
            st.write("- Повысить скорость работы кранов (модернизация)")
            st.write("- Внедрить автоматизацию погрузки")

        elif bottleneck == 'Склады':
            st.warning("🔴 **Главное узкое место: Склады**")
            st.write("Рекомендации:")
            st.write("- Увеличить емкость складов в 1,5-2 раза")
            st.write("- Ускорить вывоз грузов через ЖД")
            st.write("- Рассмотреть прямую перевалку (без складирования)")

        elif bottleneck == 'ЖД':
            st.warning("🔴 **Главное узкое место: ЖД инфраструктура**")
            st.write("Рекомендации:")
            st.write("- Увеличить количество поездов до 15-20 в сутки")
            st.write("- Добавить ЖД пути (до 5-7)")
            st.write("- Увеличить вместимость составов")

        # Детальные метрики
        st.markdown("---")
        st.subheader("📋 Детальные метрики")

        detailed_metrics = {
            "Метрика": [
                "Обработано грузов (млн т/год)",
                "Теоретическая мощность (млн т/год)",
                "Загрузка системы (%)",
                "Резерв мощности (млн т/год)",
                "Обработано судов",
                "Среднее время ожидания (ч)",
                "Среднее время обработки (ч)",
                "Средняя очередь судов",
                "Максимальная очередь судов",
                "Загрузка причалов (%)",
                "Загрузка кранов (%)",
                "Загрузка складов (%)"
            ],
            "Значение": [
                f"{results['annual_cargo_tons']/1_000_000:.2f}",
                f"{results['theoretical_max_capacity_tons']/1_000_000:.2f}",
                f"{results['capacity_utilization_pct']:.1f}",
                f"{results['reserve_capacity_tons']/1_000_000:.2f}",
                f"{results['ships_processed']}",
                f"{results['avg_wait_time_hours']:.1f}",
                f"{results['avg_processing_time_hours']:.1f}",
                f"{results['avg_queue_length']:.1f}",
                f"{int(results['max_queue_length'])}",
                f"{results['berth_utilization_pct']:.1f}",
                f"{results['crane_utilization_pct']:.1f}",
                f"{results['storage_utilization_pct']:.1f}"
            ]
        }

        st.table(pd.DataFrame(detailed_metrics))

    # Отображение анализа точки коллапса
    if 'collapse_data' in st.session_state:
        st.markdown("---")
        st.header("🔥 Анализ точки коллапса")

        collapse_data = st.session_state['collapse_data']

        col1, col2, col3 = st.columns(3)

        with col1:
            if collapse_data['first_issues_load']:
                st.metric(
                    "Первые серьезные сбои",
                    f"{collapse_data['first_issues_load']:.1f}x",
                    f"{(collapse_data['first_issues_load'] - 1) * 100:.0f}% роста"
                )
            else:
                st.success("Сбоев не обнаружено")

        with col2:
            if collapse_data['constant_delays_load']:
                st.metric(
                    "Постоянные задержки",
                    f"{collapse_data['constant_delays_load']:.1f}x",
                    f"{(collapse_data['constant_delays_load'] - 1) * 100:.0f}% роста"
                )
            else:
                st.success("Задержек не обнаружено")

        with col3:
            if collapse_data['full_collapse_load']:
                st.metric(
                    "Полный коллапс",
                    f"{collapse_data['full_collapse_load']:.1f}x",
                    f"{(collapse_data['full_collapse_load'] - 1) * 100:.0f}% роста"
                )
            else:
                st.success("Коллапса не обнаружено")

        # График зависимости метрик от нагрузки
        collapse_df = pd.DataFrame(collapse_data['results'])

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                collapse_df,
                x='load_multiplier',
                y='max_queue',
                title='Максимальная очередь от нагрузки',
                markers=True
            )
            fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Критический уровень")
            fig.update_xaxes(title='Множитель нагрузки')
            fig.update_yaxes(title='Максимальная очередь (судов)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                collapse_df,
                x='load_multiplier',
                y='avg_wait_time',
                title='Среднее время ожидания от нагрузки',
                markers=True
            )
            fig.add_hline(y=72, line_dash="dash", line_color="red", annotation_text="Критический уровень")
            fig.update_xaxes(title='Множитель нагрузки')
            fig.update_yaxes(title='Среднее время ожидания (ч)')
            st.plotly_chart(fig, use_container_width=True)

        # Таблица детальных результатов
        st.subheader("Детальные результаты анализа")
        display_df = collapse_df[['load_multiplier', 'annual_cargo_tons', 'max_queue', 'avg_wait_time', 'storage_usage', 'critical_count']].copy()
        display_df['annual_cargo_tons'] = (display_df['annual_cargo_tons'] / 1_000_000).round(2)
        display_df.columns = ['Нагрузка (x)', 'Грузооборот (млн т)', 'Макс очередь', 'Ср. ожидание (ч)', 'Склады (%)', 'Критич. показатели']
        st.dataframe(display_df, use_container_width=True)

    # Отображение результатов стресс-тестирования
    if 'stress_scenarios' in st.session_state:
        st.markdown("---")
        st.header("⚡ Результаты стресс-тестирования")

        stress_scenarios = st.session_state['stress_scenarios']

        # Сводная таблица всех сценариев
        scenario_summary = []
        for key, scenario in stress_scenarios.items():
            metrics = scenario['metrics']
            scenario_summary.append({
                'Сценарий': scenario['name'],
                'Грузооборот (млн т)': f"{metrics['annual_cargo_tons']/1_000_000:.2f}",
                'Загрузка системы (%)': f"{metrics['capacity_utilization_pct']:.1f}",
                'Макс. очередь': f"{int(metrics['max_queue_length'])}",
                'Ср. ожидание (ч)': f"{metrics['avg_wait_time_hours']:.1f}",
                'Склады (%)': f"{metrics['storage_utilization_pct']:.1f}",
                'Статус': metrics['system_status']
            })

        st.dataframe(pd.DataFrame(scenario_summary), use_container_width=True)

        # Детальный анализ каждого сценария
        st.subheader("Детальный анализ сценариев")

        for key, scenario in stress_scenarios.items():
            with st.expander(f"📊 {scenario['name']}", expanded=False):
                metrics = scenario['metrics']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Грузооборот",
                        f"{metrics['annual_cargo_tons']/1_000_000:.2f} млн т"
                    )

                with col2:
                    color = "🟢" if metrics['capacity_utilization_pct'] < 60 else "🟡" if metrics['capacity_utilization_pct'] < 80 else "🔴"
                    st.metric(
                        f"{color} Загрузка системы",
                        f"{metrics['capacity_utilization_pct']:.1f}%"
                    )

                with col3:
                    color = "🟢" if metrics['avg_wait_time_hours'] < 24 else "🟡" if metrics['avg_wait_time_hours'] < 72 else "🔴"
                    st.metric(
                        f"{color} Ср. ожидание",
                        f"{metrics['avg_wait_time_hours']:.1f} ч"
                    )

                with col4:
                    color = "🟢" if metrics['max_queue_length'] < 5 else "🟡" if metrics['max_queue_length'] < 10 else "🔴"
                    st.metric(
                        f"{color} Макс. очередь",
                        f"{int(metrics['max_queue_length'])} судов"
                    )

                # Критические проблемы
                if metrics['critical_issues']:
                    st.error("**Критические проблемы:**")
                    for issue in metrics['critical_issues']:
                        st.write(f"- {issue}")
                else:
                    st.success("✅ Критических проблем не обнаружено")

                # Узкое место
                st.info(f"**Узкое место:** {metrics['bottleneck']}")

    else:
        st.info("👈 Настройте параметры в боковой панели и нажмите **Запустить симуляцию**")


if __name__ == "__main__":
    main()
