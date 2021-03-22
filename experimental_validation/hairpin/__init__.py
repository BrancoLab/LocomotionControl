from experimental_validation.hairpin.dbase.djconn import start_connection

dbname, schema = start_connection()

print(f"Connected to database: {dbname}")
