import psycopg2
import pandas as pd
import openpyxl

workbook = openpyxl.load_workbook("medicine.xlsx")
worksheet = workbook['easy']
data = []

#Получаем данные из Excel
for row in worksheet.iter_rows(min_row=2, values_only=True): 
    data.append({"id" : row[0], "analysis_id" : row[1], "value" : row[2]})
workbook.close()
# print(data)

# Подключаемся к PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432")

#создаем курсор
cursor = conn.cursor()

# cursor.execute("TRUNCATE med_an_name RESTART IDENTITY")
# cursor.execute("TRUNCATE med_name RESTART IDENTITY")

# # заполняем med_an_name и med_name
# cursor.execute("""
#     INSERT INTO med_an_name (id, name, is_simple, min_value, max_value) VALUES
#         ('IG', 'Иммуноглобулин общий', 'N', 0.00, 100.00),
#         ('ALAT', 'Аланинаминотрансфераза ', 'N', 0.00, 45.00),
#         ('2-A', 'Лейкоциты (моча)', 'N', 0.00, 5.00),
#         ('G124', 'Гаптоглобин', 'N', 150.00, 2000.00),
#         ('890', 'Глюкоза', 'N', 3.30, 5.50),
#         ('S', 'Щелочная фосфатаза', 'N', 40.00, 150.00),
#         ('1-100', 'Общий белок ', 'N', 64.00, 84.00),
#         ('N', 'Нитрит (моча)', 'Y', NULL, NULL),
#         ('3-511', 'Креатинин', 'N', 62.00, 115.00),
#         ('Z', 'Эритроциты (моча)', 'N', 0.00, 2.00),
#         ('IRR', 'Липаза', 'N', 0.00, 190.00),
#         ('GLK', 'Глюкоза (моча)', 'Y', NULL, NULL),
#         ('1-875', 'Грибки (моча)', 'Y', NULL, NULL),
#         ('1-900', 'Амилаза панкреатическая', 'N', 0.00, 50.00),
#         ('BBB', 'Белок (моча)', 'Y', NULL, NULL),
#         ('AU', 'Гематокрит', 'N', 39.00, 49.00),
#         ('C', 'СОЭ', 'N', 2.00, 20.00),
#         ('aZz', 'Билирубин', 'N', 5.00, 20.00);
# """)

# cursor.execute("""
#     INSERT INTO med_name (id, name, phone) VALUES
#         (191, 'Чемиренко Д.А', '+7 (905) 912-28-47'),
#         (140, 'Головцов Р.Р', '+7 (906) 949-98-17'),
#         (119, 'Сакобов А.Т', '+7 (952) 996-60-21'),
#         (57, 'Тарлов И.Е', '+7 (900) 875-67-38'),
#         (195, 'Свенюков Б.Ю', '+7 (983) 577-27-21'),
#         (53, 'Попелицкий Р.Г', '+7 (976) 841-79-95'),
#         (96, 'Умралиев Р.Э', '+7 (941) 954-55-88'),
#         (62, 'Бондарев С.Г', '+7 (906) 933-95-63'),
#         (199, 'Цетнарский Э.И', '+7 (908) 245-80-47'),
#         (114, 'Даклеев Т.М', '+7 (925) 612-93-95'),
#         (73, 'Гармокацкий С.Л', '+7 (935) 653-32-78'),
#         (80, 'Маржецкий А.Я', '+7 (949) 912-97-84'),
#         (151, 'Поляховский Г.Д', '+7 (998) 596-45-47'),
#         (127, 'Галендук Л.Г', '+7 (935) 164-87-46'),
#         (135, 'Номоконов Г.Р', '+7 (940) 384-74-90'),
#         (77, 'Шелкоплясов Э.В', '+7 (979) 643-71-75'),
#         (67, 'Эльменькин Д.Р', '+7 (982) 643-36-73'),
#         (176, 'Ламок А.О', '+7 (984) 779-54-57'),
#         (163, 'Бочечкаров Э.Д', '+7 (943) 102-79-69'),
#         (192, 'Голяминских Г.Р', '+7 (903) 280-60-58'),
#         (72, 'Пахарев Ю.Э', '+7 (977) 810-48-75'),
#         (111, 'Ведмидев Я.У', '+7 (997) 541-22-17'),
#         (118, 'Сисемкин В.Р', '+7 (960) 542-96-83'),
#         (148, 'Штыкулин И.А', '+7 (949) 862-47-83'),
#         (162, 'Шапаев Л.А', '+7 (919) 109-43-56'),
#         (76, 'Млицкий С.А', '+7 (987) 542-73-38'),
#         (101, 'Загатин А.Д', '+7 (987) 928-64-89');
# """)

# # Сохраняем изменения
# conn.commit()

# Берем данные из таблиц med_an_name и med_name и преобразуем их в DataFrame
cursor.execute("SELECT * FROM med_an_name")
rows_analyse = cursor.fetchall()
columns_analyse = []
for desc in cursor.description:
    columns_analyse.append(desc[0])
df_analyse = pd.DataFrame(rows_analyse, columns=columns_analyse)

cursor.execute("SELECT * FROM med_name")
rows_patients = cursor.fetchall()
columns_patients = []
for desc in cursor.description:
    columns_patients.append(desc[0])
df_patients = pd.DataFrame(rows_patients, columns=columns_patients)

# Закрываем соединение с БД
cursor.close()
conn.close()

# print(df_analyse)
# print(df_patients)

analyse_dict = df_analyse.set_index("id").to_dict("index")
patients_dict = df_patients.set_index("id").to_dict("index")

output = []
for row in data:

    # получаем id пациента, id анализа и значение
    patient_id = row["id"]
    analyse_id = row["analysis_id"]
    value = row["value"]

    # Получаем информацию о пациенте и анализе
    patient = patients_dict.get(patient_id)
    analyse_info = analyse_dict.get(analyse_id)

    # Проверяем, что данные существуют
    if not patient or not analyse_info:
        continue

    result = ""

    # Проверяем, является ли анализ простым
    is_simple = analyse_info.get("is_simple", "").strip() == "Y"

    if is_simple:
        # Если анализ простой, проверяем значение
        if str(value).strip() == "+":
            result = "Положительный"
    else:
        # Если анализ не простой, проверяем числовые значения
        try:
            value = float(value)
            min_value = analyse_info.get("min_value")
            max_value = analyse_info.get("max_value")

            if min_value is not None and value < min_value:
                result = "Понижен"
            elif max_value is not None and value > max_value:
                result = "Повышен"
        except ValueError:
            continue  # Пропускаем, если значение не числовое

    # Если результат найден, добавляем в выходной список
    if result:
        output.append({
            "ФИО": patient.get("name"),
            "Телефон": patient.get("phone"),
            "Анализ": analyse_info.get("name"),
            "Заключение": result
        })

df_out = pd.DataFrame(output)
df_out.to_excel("result_easy.xlsx", index=False)
print("Результат сохранен в result_easy.xlsx")