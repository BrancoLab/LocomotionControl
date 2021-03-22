import datajoint as dj

ip = "127.0.0.1"  # "localhost"

psw = "fede"


def start_connection():
    """
    Starts up the datajoint database and signs in with user and passoword + returns the database name
    
    docker compose yaml file:
    D:\Dropbox (UCL - SWC)\Rotation_vte\mysql-server\docker-compose.yml

    Data are here:
    D:\Dropbox (UCL - SWC)\Rotation_vte\mysql-server\data\Database

    """
    dbname = "DatabaseV4"  # Name of the database subfolder with data
    if dj.config["database.user"] != "root":

        dj.config["database.host"] = ip
        dj.config["database.user"] = "root"
        dj.config["database.password"] = psw
        dj.config["database.safemode"] = True
        dj.config["safemode"] = False
        dj.config["enable_python_native_blobs"] = True

        dj.conn()

    schema = dj.schema(dbname)
    return dbname, schema


def print_erd():
    _, schema = start_connection()
    dj.ERD(schema).draw()


if __name__ == "__main__":
    start_connection()
    print_erd()
