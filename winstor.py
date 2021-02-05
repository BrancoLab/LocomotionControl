from control.manager import Manager
import click
from pyinspect import install_traceback

install_traceback(
    keep_frames=5, all_locals=True, relevant_only=False,
)


@click.command()
@click.option("--trialn", default=None, help="Trial number.")
@click.option("--config", default=None, help="Config.json file")
def main(trialn, config):
    if trialn is not None:
        trialn = int(trialn)
    Manager(winstor=True, trialn=trialn, config_file=config, to_db=True).run(
        n_secs=12
    )


if __name__ == "__main__":
    main()
