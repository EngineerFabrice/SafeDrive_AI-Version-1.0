import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='safedrive_ai', port=3307)
cursor = conn.cursor()
cursor.execute('DESCRIBE trips')
cols = cursor.fetchall()

print("Trips table columns:")
print("-" * 50)
for col in cols:
    col_name, col_type = col[0], col[1]
    print(f"{col_name}: {col_type}")

cursor.close()
conn.close()
