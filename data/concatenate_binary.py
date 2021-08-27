from loguru import logger
from pathlib import Path

from fcutils.path import size
from fcutils.progress import progess_with_description as progress


"""
    Code to concatenate two large binary files
"""


def concatenate(source1: Path, source2: Path, dest: Path):
    """
        It concatenates two large binary files into an even larger one
    """
    chunk_size: int = 2048

    # create destination folder
    if not dest.parent.exists():
        dest.parent.parent.mkdir()
        dest.parent.mkdir()

    # add logs
    logger.add(dest.parent / "logs.logs")

    logger.info(
        f'About to merge: "{source1.name}" ({size(source1)}) and "{source2.name}" ({size(source2)})'
    )
    logger.info(f"Saving merged file at: {dest}")

    # open writing file
    progress.start()
    fout = open(dest, "ab+")
    for source in (source1, source2):
        logger.info(f'Processing: "{source.name}"')
        tid = progress.add_task(
            f'Processing "{source.name}"', total=size(source, fmt=False)
        )

        fin = open(source, "rb")
        # read by chunk
        while True:
            chunk = fin.read(chunk_size)
            if len(chunk) < chunk_size:
                raise ValueError(
                    f"Got chunk of length: {len(chunk)}, expected {chunk_size} bytes"
                )

            if not chunk:
                # move on to the next
                break

            # write to destination file
            fout.write(chunk)
            progress.update(tid, advance=chunk_size)

        fin.close()
        progress.remove_task(tid)
    progress.stop()
    fout.close()


if __name__ == "__main__":
    recs_fld = Path(r"W:\swc\branco\Federico\Locomotion\raw\recordings")

    r1 = "210714_750_longcol_intref_hairpin_g0"
    r2 = "210714_750_longcol_intref_openarena_g0"
    d = "210714_750_longcol_intref_CONCATENATED_g0"

    s1 = (
        recs_fld
        / Path(r1)
        / Path(r1 + "_imec0")
        / Path(r1 + "_t0.imec0.ap.bin")
    )
    s2 = (
        recs_fld
        / Path(r2)
        / Path(r2 + "_imec0")
        / Path(r2 + "_t0.imec0.ap.bin")
    )
    dest = (
        recs_fld / Path(d) / Path(d + "_imec0") / Path(d + "_t0.imec0.ap.bin")
    )

    concatenate(s1, s2, dest)
