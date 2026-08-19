import duckdb

con = duckdb.connect("strategiz.db")

tables = con.execute("SHOW TABLES").fetchall()

print("Tables in database:", tables)