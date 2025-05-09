from src.data_collector import DataCollector
from src.data_analyzer import DataAnalyzer
from src.model_trainer import TaxiModel
from src.preprocessing_pipeline import FeatureEngineer

import pandas as pd
import os

collector = None
engineer = FeatureEngineer()
model = None
latest_model_path = "models/latest_model.pkl"

def main():
    global collector, model
    while True:
        command = input("\nВведите команду (Init / Update / Inference / Summary / Exit): ").strip()

        if command.lower() == 'exit':
            print("Завершение работы.")
            break

        elif command.lower() == 'init':
            try:
                initial_size = int(input("Размер стартовых данных: "))
                collector = DataCollector("train.csv", initial_size)
                print(f"Инициализировано {initial_size} строк.")
            except Exception as e:
                print(f"Ошибка инициализации: {e}")

        elif command.lower() == 'update':
            if not os.path.exists("models"):
                os.mkdir("models")
            if collector is None:
                print("Сначала выполните Init.")
                continue
            try:
                batch_size = int(input("Размер батча: "))
                model_type = str(input("Модель (LR, KNN, DT, RF, Lasso, Ridge): "))
                is_warm_start = bool(int(input("Дообучаем модель, если возможно? (0 или 1): ")))

                model = TaxiModel(model_type)
                if model.pipeline == 0:
                    continue
                new_data = collector.get_batch(batch_size)
                analyzer = DataAnalyzer(new_data)
                clean_df = analyzer.fit_transform()

                target = clean_df['log_trip_duration']
                features = clean_df.drop(columns=["log_trip_duration"])

                model.train(model_type, features, target, is_warm_start)
                model.save(latest_model_path)
                model.save(f"models/{model_type}_model.pkl")
                print(f"Модель обучена на {len(clean_df)} строках и сохранена как {latest_model_path}.")
            except Exception as e:
                print(f"Ошибка во время обновления: {e}")

        elif command.lower() == 'inference':
            try:
                path = input("Путь к CSV с входными данными: ")
                model_name = input("Имя модели (например, latest): ").strip()
                metric = str(input("Метрика (RMSE, MAE): "))

                if not os.path.exists(path):
                    print("Файл не найден.")
                    continue

                infer_model = TaxiModel.load(f"models/{model_name}_model.pkl")
                df = pd.read_csv(path)

                analyzer = DataAnalyzer(df)
                clean_df = analyzer.fit_transform()

                true_values = clean_df['log_trip_duration']
                features = clean_df.drop(columns=["log_trip_duration"])
                result = infer_model.predict(features, true_values, metric)

                print(f"MAE: {result:.2f} секунд на {len(true_values)} примерах")
            except Exception as e:
                print(f"Ошибка инференса: {e}")

        elif command.lower() == 'summary':
            if collector is None:
                print("Сначала выполните Init.")
                continue
            try:
                analyzer = DataAnalyzer(collector.batch_data)
                stats = analyzer._calculate_basic_stats()
                print("Статистика по текущей выборке:")
                for k, v in stats.items():
                    print(f"{k} :")
                    for w, ans in v.items():
                        print(f"\t{w} : {ans}")
                    print()
            except Exception as e:
                print(f"Ошибка при расчёте статистики: {e}")

        else:
            print("Неизвестная команда. Попробуйте снова.")

if __name__ == "__main__":
    main()

