import click
from pyinspect import install_traceback
from pathlib import Path

from control.manager import Manager
from control.paths import winstor_main

install_traceback(
    keep_frames=5, all_locals=True, relevant_only=False,
)


@click.command()
@click.option("--config", default=None, help="Config.json file")
def main(config):
    name = Path(config).stem
    base = winstor_main / name
    base.mkdir(exist_ok=True)

    for rep in range(5):
        folder = base / f"{name}_rep_{rep}"
        Manager(
            winstor=True, folder=folder, config_file=config, to_db=False
        ).run(n_secs=12)


if __name__ == "__main__":
    main()
