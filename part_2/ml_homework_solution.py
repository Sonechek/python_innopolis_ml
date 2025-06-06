# Домашняя работа 1: Машинное обучение
# Решение задачи предсказания стоимости недвижимости

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных (предполагаем, что файл уже загружен)
# !gdown 1yQgwqFxwkHtZL2PZ2waF4Pg3Sb9hyHo9
# df = pd.read_csv('df_with_nan.csv')

# Для демонстрации создадим тестовые данные с пропусками
# В реальном решении используйте загруженные данные
np.random.seed(42)
n_samples = 20640

# Создаем данные, похожие на California Housing dataset
data = {
    'MedInc': np.random.normal(6.0, 2.0, n_samples),
    'HouseAge': np.random.uniform(1, 52, n_samples),
    'AveRooms': np.random.normal(6.0, 2.0, n_samples),
    'AveBedrms': np.random.normal(1.0, 0.5, n_samples),
    'Population': np.random.uniform(3, 35000, n_samples),
    'AveOccup': np.random.normal(3.0, 2.0, n_samples)
}

df = pd.DataFrame(data)
# Создаем целевую переменную
df['Target'] = (df['MedInc'] * 0.4 + 
                df['AveRooms'] * 0.1 - 
                df['HouseAge'] * 0.01 + 
                np.random.normal(0, 0.5, n_samples))

# Добавляем пропуски случайным образом
missing_mask = np.random.random(df.shape) < 0.05  # 5% пропусков
df = df.mask(missing_mask)

print("="*60)
print("ДОМАШНЯЯ РАБОТА 1: МАШИННОЕ ОБУЧЕНИЕ")
print("ТЕМА: Предсказание стоимости недвижимости")
print("="*60)

# ==========================================
# 1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
# ==========================================

print("\n" + "="*50)
print("1. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print("="*50)

# Размер датасета
print("\n📊 РАЗМЕР ДАТАСЕТА:")
print(f"Размер в памяти: {df.memory_usage(deep=True).sum() / (1024**3):.6f} ГБ")
print(f"Количество строк: {df.shape[0]:,}")
print(f"Количество столбцов: {df.shape[1]}")

# Разделение на признаки и целевую переменную
X = df.drop('Target', axis=1)
y = df['Target']

# Анализ числовых признаков
print("\n📈 СТАТИСТИЧЕСКИЙ АНАЛИЗ ПРИЗНАКОВ:")
print("-" * 80)
stats_df = pd.DataFrame({
    'Признак': X.columns,
    'Минимум': X.min().values,
    'Максимум': X.max().values,
    'Среднее': X.mean().values,
    'Медиана': X.median().values,
    'Стд. отклонение': X.std().values
})
print(stats_df.round(3).to_string(index=False))

# Анализ пропусков
print("\n🔍 АНАЛИЗ ПРОПУСКОВ:")
print("-" * 40)
missing_info = pd.DataFrame({
    'Признак': df.columns,
    'Количество пропусков': df.isnull().sum().values,
    'Процент пропусков': (df.isnull().sum() / len(df) * 100).values
})
missing_info = missing_info[missing_info['Количество пропусков'] > 0]
if len(missing_info) > 0:
    print(missing_info.round(2).to_string(index=False))
else:
    print("Пропусков не обнаружено!")

# ==========================================
# 2. РАЗДЕЛЕНИЕ ДАННЫХ НА ТРЕЙН И ТЕСТ
# ==========================================

print("\n" + "="*50)
print("2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape[0]:,} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]:,} образцов")

# ==========================================
# 3. ОБРАБОТКА ПРОПУСКОВ
# ==========================================

print("\n" + "="*50)
print("3. ПОДГОТОВКА ДАННЫХ К ОБУЧЕНИЮ")
print("="*50)

print("\n🔧 ОБРАБОТКА ПРОПУСКОВ:")

# Способ 1: Заполнение средним значением
print("\n1️⃣ СПОСОБ 1: Заполнение средним значением")
print("   Обоснование: Простой и эффективный метод для числовых данных,")
print("   не изменяет общее распределение данных значительно.")

imputer_mean = SimpleImputer(strategy='mean')
X_train_mean = pd.DataFrame(
    imputer_mean.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_mean = pd.DataFrame(
    imputer_mean.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Способ 2: Заполнение медианой
print("\n2️⃣ СПОСОБ 2: Заполнение медианой")
print("   Обоснование: Более устойчив к выбросам, чем среднее значение,")
print("   лучше подходит для данных с асимметричным распределением.")

imputer_median = SimpleImputer(strategy='median')
X_train_median = pd.DataFrame(
    imputer_median.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_median = pd.DataFrame(
    imputer_median.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ==========================================
# 4. ПОСТРОЕНИЕ МОДЕЛЕЙ
# ==========================================

print("\n" + "="*50)
print("4. ПОСТРОЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
print("="*50)

# Функция для оценки модели
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Модель': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

results = []

# Модель 1: Линейная регрессия с заполнением средним
print("\n🤖 МОДЕЛЬ 1: Линейная регрессия + заполнение средним")
lr_mean = LinearRegression()
lr_mean.fit(X_train_mean, y_train)
y_pred_mean = lr_mean.predict(X_test_mean)
results.append(evaluate_model(y_test, y_pred_mean, "LinearRegression + Mean"))

# Модель 2: Линейная регрессия с заполнением медианой
print("🤖 МОДЕЛЬ 2: Линейная регрессия + заполнение медианой")
lr_median = LinearRegression()
lr_median.fit(X_train_median, y_train)
y_pred_median = lr_median.predict(X_test_median)
results.append(evaluate_model(y_test, y_pred_median, "LinearRegression + Median"))

# ==========================================
# 5. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ
# ==========================================

print("\n" + "="*50)
print("5. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("="*50)

results_df = pd.DataFrame(results)
print("\n📊 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛЕЙ:")
print("-" * 70)
print(results_df.round(4).to_string(index=False))

# Определение лучшей модели
best_model_idx = results_df['R²'].idxmax()
best_model = results_df.iloc[best_model_idx]

print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model['Модель']}")
print(f"   R² Score: {best_model['R²']:.4f}")
print(f"   RMSE: {best_model['RMSE']:.4f}")

# ==========================================
# 6. АНАЛИЗ КОЭФФИЦИЕНТОВ
# ==========================================

print("\n" + "="*50)
print("6. АНАЛИЗ КОЭФФИЦИЕНТОВ ЛУЧШЕЙ МОДЕЛИ")
print("="*50)

# Выбираем лучшую модель
if best_model['Модель'] == "LinearRegression + Mean":
    best_lr_model = lr_mean
    feature_names = X_train_mean.columns
else:
    best_lr_model = lr_median
    feature_names = X_train_median.columns

coefficients_df = pd.DataFrame({
    'Признак': feature_names,
    'Коэффициент': best_lr_model.coef_,
    'Важность': np.abs(best_lr_model.coef_)
}).sort_values('Важность', ascending=False)

print("\n📈 КОЭФФИЦИЕНТЫ МОДЕЛИ:")
print("-" * 40)
print(coefficients_df.round(4).to_string(index=False))
print(f"\nСвободный член (intercept): {best_lr_model.intercept_:.4f}")

# ==========================================
# 7. ВЫВОДЫ И РЕКОМЕНДАЦИИ
# ==========================================

print("\n" + "="*60)
print("7. ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*60)

print("\n📋 ОСНОВНЫЕ ВЫВОДЫ:")
print("-" * 30)

print(f"""
1. 📊 РАЗМЕР ДАННЫХ:
   • Датасет содержит {df.shape[0]:,} записей и {df.shape[1]} признаков
   • Размер в памяти: {df.memory_usage(deep=True).sum() / (1024**3):.6f} ГБ

2. 🔍 ПРОПУСКИ В ДАННЫХ:
   • Обнаружены пропуски в {missing_info.shape[0] if len(missing_info) > 0 else 0} признаках
   • Применены два метода заполнения: среднее и медиана

3. 🤖 КАЧЕСТВО МОДЕЛЕЙ:
   • Лучшая модель: {best_model['Модель']}
   • R² Score: {best_model['R²']:.4f} (объясняет {best_model['R²']*100:.1f}% дисперсии)
   • RMSE: {best_model['RMSE']:.4f}

4. 📈 ВАЖНЫЕ ПРИЗНАКИ:
   • Наиболее влиятельный: {coefficients_df.iloc[0]['Признак']} (коэф. {coefficients_df.iloc[0]['Коэффициент']:.4f})
   • Наименее влиятельный: {coefficients_df.iloc[-1]['Признак']} (коэф. {coefficients_df.iloc[-1]['Коэффициент']:.4f})
""")

print("\n💡 РЕКОМЕНДАЦИИ:")
print("-" * 20)

recommendation = "заполнение средним" if "Mean" in best_model['Модель'] else "заполнение медианой"
print(f"""
• РЕКОМЕНДУЕТСЯ использовать модель LinearRegression с методом "{recommendation}"
• Причины выбора:
  - Показывает наилучшее качество по метрике R² ({best_model['R²']:.4f})
  - Имеет наименьшую ошибку RMSE ({best_model['RMSE']:.4f})
  - Простая и интерпретируемая модель
  
• Дальнейшие улучшения:
  - Можно попробовать полиномиальные признаки
  - Применить регуляризацию (Ridge, Lasso)
  - Использовать более сложные методы заполнения пропусков
""")

print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЕН ✅")
print("="*60)