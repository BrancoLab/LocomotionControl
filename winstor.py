from control.manager import Manager
import click
from pyinspect import install_traceback

install_traceback(
    keep_frames=5, all_locals=True, relevant_only=False,
)


@click.command()
@click.option("--trialn", default=None, help="Trial number.")
def main(trialn):
    Manager(winstor=False, trialn=trialn).run(n_secs=10)


if __name__ == "__main__":
    main()
