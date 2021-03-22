from experimental_validation.hairpin.dbase.djconn import start_connection

dbaname, schema = start_connection()

print(f"Connected to database: {dbname}")
