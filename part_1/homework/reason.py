import psycopg2
import re


# подключение к базе данных и получение данных
def get_reasons_from_db():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432")

    cursor = conn.cursor()
    
    cursor.execute("SELECT reason, reason_correct FROM payments LIMIT 10")
    reasons = cursor.fetchall()

    cursor.close()
    conn.close()

    return reasons


# преобразование строки reason
def conversion_reason(reason):
    reason = re.sub(r'\\', '/', reason) # установка правильного слэша
    reason = re.sub(r'\s+', ' ', reason) # удаление лишних пробелов
    reason = re.sub(r'^\s+|\s+$', '', reason) # удаление пробелов в начале и конце строки
    reason = re.sub(r'([0-9]{2})\s*\.\s*([0-9]{2})\s*\.\s*([0-9]{4})', r'\1.\2.\3', reason) # удаление разрыва в дате
    reason = re.sub(r'(\d+/\d+)\s+(\d{2}\.\d{2}\.\d{4})', r'\1 от \2', reason) # вставить "от" между договором и датой
    
    return reason


# подсчет результатов
def counting_reasons():
    reasons = get_reasons_from_db()
    summary = 0
    total = len(reasons)
    for reason, reason_correct in reasons:
        conversion = conversion_reason(reason)
        # print(conversion, reason_correct)
        if conversion == reason_correct:
            summary += 1

    print(f"Совпало: {summary} / {total} ({summary / total * 100:.2f}%)")


counting_reasons()
