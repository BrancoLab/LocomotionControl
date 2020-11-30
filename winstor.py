from control.manager import Manager
import click


@click.command()
@click.option("--trialn", default=0, help="Trial number.")
def main(trialn):
    Manager(winstor=True, trialn=trialn).run()


if __name__ == "__main__":
    main()
