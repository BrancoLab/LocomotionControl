from proj import (
    Model,
    Environment,
    Controller,
    run_experiment,
)
import click


@click.command()
@click.option("--trialn", default=0, help="Trial number.")
def main(trialn):
    model = Model(trial_n=trialn)
    env = Environment(model, winstor=True)
    control = Controller(model)
    run_experiment(env, control, model)


if __name__ == "__main__":
    main()
