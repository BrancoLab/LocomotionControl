from loguru import logger

from data.dbase.djconn import start_connection

try:
    dbname, schema = start_connection()
except Exception as e:
    logger.warning(f"Failed to connect to datajoint database:\n{e}")
    connected = False
else:
    logger.debug(f"Connected to database: {dbname}")
    connected = True

from data.dbase import db_tables
