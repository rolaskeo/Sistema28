import sqlite3

db_path = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
sql_path = r"C:\Rolex\Python\Sistema28Script\migrations\001_migrate_v30.sql"

con = sqlite3.connect(db_path)
with open(sql_path, "r", encoding="utf-8") as f:
    sql_script = f.read()
con.executescript(sql_script)
con.close()

print("Migración aplicada con éxito ✅")
