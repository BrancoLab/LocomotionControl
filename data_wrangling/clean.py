from fcutils.progress import track

from fcutils.path import files


for f in track(
    files(r"Z:\swc\branco\Federico\Locomotion\control\behavioural_data\philip")
):
    clean_name = (
        f.stem.replace(" ", "")
        .replace("'", "")
        .replace(".", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("-", "_")
    )
    new = f.parent / (clean_name + f.suffix)
    f.rename(new)
    a = 1
