import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict


class MarketDataProvider:
    """Сервис для получения исторических рыночных данных."""

    @staticmethod
    def get_historical_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
        """
        Загружает скорректированные цены закрытия (Adj Close) с Yahoo Finance.

        :param tickers: Список тикеров активов (например, ['SPY', 'TLT', 'GLD']).
        :param period: Период загрузки данных (по умолчанию за 5 лет).
        :return: DataFrame со скорректированными ценами закрытия.
        """
        if not tickers:
            raise ValueError("Список тикеров пуст. Невозможно загрузить данные.")
 
        try:
            print(f"Загрузка данных для {tickers} за период {period}...")
            # Загружаем данные. Если тикер один, yfinance возвращает Series,
            # если несколько - DataFrame. Обрабатываем это.
            data = yf.download(tickers, period=period, progress=False)

            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            elif len(tickers) == 1:
                prices = pd.DataFrame(data['Adj Close'])
                prices.columns = tickers
            else:
                raise ValueError("Неожиданный формат данных от yfinance")

            # Очистка данных: удаляем столбцы, где все значения NaN
            prices = prices.dropna(axis=1, how='all')
            # Заполняем пропуски предыдущими значениями (forward fill)
            prices = prices.fillna(method='ffill')
            # Если остались NaN в начале - удаляем эти строки
            prices = prices.dropna()

            print(f"Данные успешно загружены. Формат: {prices.shape}")
            return prices

        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузге данных с Yahoo Finance: {e}")


class OptimizerService:
    """Математическое ядро системы для портфельной оптимизации."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Инициализация сервиса оптимизации.
        :param risk_free_rate: Безрисковая процентная ставка (по умолчанию 2%).
        """
        self.risk_free_rate = risk_free_rate

    def calculate_optimal_weights(self, prices_df: pd.DataFrame) -> Dict[str, float]:
        """
        Реализует поиск оптимальных весов портфеля по модели Марковица
        (максимизация коэффициента Шарпа).

        :param prices_df: DataFrame с историческими ценами закрытия.
        :return: Словарь вида {'Тикер': Вес_в_процентах}.
        """
        if prices_df.empty:
            raise ValueError("Получен пустой DataFrame для оптимизации.")

        # 1. Расчет ожидаемой годовой доходности и ковариационной матрицы
        # Считаем ежедневные логарифмические доходности
        log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

        # 252 торговых дня в году
        mean_returns = log_returns.mean() * 252
        cov_matrix = log_returns.cov() * 252

        num_assets = len(prices_df.columns)
        tickers = prices_df.columns.tolist()

        # 2. Определяем целевую функцию (Минимизируем отрицательный Шарп)
        def negative_sharpe(weights: np.array) -> float:
            # Ожидаемая доходность портфеля
            p_ret = np.sum(mean_returns * weights)
            # Ожидаемая волатильность (риск) портфеля
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Коэффициент Шарпа
            sharpe_ratio = (p_ret - self.risk_free_rate) / p_vol
            # Возвращаем отрицательное значение, т.к. функция scipy минимизирует
            return -sharpe_ratio

        # 3. Задаем ограничения и границы
        # Ограничение 1: Сумма всех весов должна быть равна 1 (100%)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

        # Ограничение 2: Вес каждого актива от 0 до 1 (запрет коротких позиций)
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        # Начальное приближение: равные доли для всех активов
        init_guess = np.array(num_assets * [1. / num_assets, ])

        # 4. Запуск оптимизатора
        print("Запуск алгоритма оптимизации (SLSQP)...")
        result = minimize(negative_sharpe,
                          init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        if result.success:
            optimal_weights = result.x
            # Формируем итоговый словарь {Тикер: Вес в %} и округляем до 2 знаков
            weights_dict = {ticker: round(weight * 100, 2)
                            for ticker, weight in zip(tickers, optimal_weights)}
            print("Оптимизация успешно завершена.")
            return weights_dict
        else:
            raise RuntimeError(f"Оптимизация не удалась. Причина: {result.message}")


# --- Пример использования (для проверки работоспособности кода) ---
if __name__ == "__main__":
    # Выбираем несколько тикеров для теста (Акции США, Облигации, Золото)
    test_tickers = ['SPY', 'TLT', 'GLD']

    try:
        # Загружаем данные
        provider = MarketDataProvider()
        prices = provider.get_historical_prices(test_tickers, period="3y")

        # Запускаем оптимизацию
        optimizer = OptimizerService(risk_free_rate=0.04)  # Текущая ставка около 4%
        optimal_allocation = optimizer.calculate_optimal_weights(prices)

        print("\n--- Оптимальное распределение портфеля ---")
        for ticker, weight in optimal_allocation.items():
            print(f"{ticker}: {weight}%")

    except Exception as err:
        print(f"\nКритическая ошибка выполнения: {err}")