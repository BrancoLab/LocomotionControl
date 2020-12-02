from control.manager import Manager
import click


@click.command()
@click.option("--trialn", default=None, help="Trial number.")
def main(trialn):
    Manager(winstor=True, trialn=trialn).run(n_secs=10)


if __name__ == "__main__":
    main()
