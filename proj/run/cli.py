import click

from proj import Model, Environment, Controller, run_experiment

@click.command()
@click.option("-f", "--folder", default=None, help="Save folder")
@click.option("-n", "--n_steps", default=1000, help="number of steps")
@click.option("-ip", "--int_plot", default=True, help="interactive plotting")

def launch(folder=None, n_steps=1000, int_plot=True):
    
    agent = Model()

    print(f'Launch locomotion simulation with config: {agent}')

    env = Environment(agent)
    control = Controller(agent)
    run_experiment(env, control, agent, n_steps=2000, plot=int_plot)

