from control.manager import Manager
import click
import shutil
from pyinspect import install_traceback
from loguru import logger


install_traceback(
    keep_frames=5, all_locals=True, relevant_only=False,
)


@click.command()
@click.option("--trialn", default=None, help="Trial number.")
@click.option("--config", default=None, help="Config.json file")
def main(trialn, config):
    if trialn is not None:
        trialn = int(trialn)

    try:
        manager = Manager(
            winstor=True, trialn=trialn, config_file=config, to_db=False
        ).run(n_secs=4)
    except Exception as e:
        print(e)
        try:
            logger.remove()
            shutil.rmtree(manager.datafolder)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
