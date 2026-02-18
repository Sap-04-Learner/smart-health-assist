import sqlite3

# Connect to the database
conn = sqlite3.connect('./data/chroma_db/chroma.sqlite3')
cursor = conn.cursor()

# Query to fetch table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print table names
for table in tables:
   print(table[0])

# Query the table
cursor.execute("SELECT * FROM acquire_write")
rows = cursor.fetchall()

# Print the contents
for row in rows:
    print(row)

conn.close()