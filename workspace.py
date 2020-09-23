# %%
from proj.utils.dropbox import DropBoxUtils

db = DropBoxUtils()


# %%
from pathlib import Path

fld = Path("/Users/federicoclaudi/Documents/Github/pysical_locomotion/proj")


# %%
def upload_folder(dbx, fld, base):
    fld = Path(fld)
    base = Path(base)

    # loop subfolders
    for subf in fld.glob("*"):
        if not subf.is_dir():
            continue
        upload_folder(dbx, subf, base / subf.name)

    for f in fld.glob("*.*"):
        if not f.is_file():
            continue

        dest = base / f.name
        dbx.upload_file(f, dest)


base = ""
upload_folder(db, fld, "test")
# %%
db.upload_folder(fld, "test2")

# %%
