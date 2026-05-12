import pymysql

# Database connection
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='safedrive_ai',
    port=3307
)

cursor = conn.cursor()

# Read and execute schema.sql
with open('schema.sql', 'r') as f:
    sql_script = f.read()

# Split by semicolon and execute each statement
statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

for statement in statements:
    if statement:
        try:
            cursor.execute(statement)
            print(f"Executed: {statement[:50]}...")
        except Exception as e:
            print(f"Error executing: {statement[:50]}... - {e}")

conn.commit()
cursor.close()
conn.close()

print("✅ Database schema setup completed!")