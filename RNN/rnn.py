import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/federicoclaudi/Documents/Github/pyrnn")

from typing import Tuple

from pyrnn import CTRNN as RNN
from pyrnn.plot import plot_model_weights
from pyrnn.connectivity import MultiRegionConnectivity, Region


def initialize_rnn(**rnn_kwargs) -> Tuple[RNN, MultiRegionConnectivity]:
    # ------------------------------ create regions ------------------------------ #
    mos = Region(
        name="mos",
        n_units=64,
        dale_ratio=0.8,  # only excitatory
        autopses=False,
    )

    stn = Region(name="stn", n_units=64, dale_ratio=0.8, autopses=False,)

    cun = Region(name="cun", n_units=64, dale_ratio=0.5, autopses=False,)

    grn_l = Region(name="grn_l", n_units=64, dale_ratio=0.8, autopses=False,)

    grn_r = Region(name="grn_r", n_units=64, dale_ratio=0.8, autopses=False,)

    # ---------------------------- create connections ---------------------------- #
    mrc = MultiRegionConnectivity(mos, stn, cun, grn_l, grn_r)

    # create feedforward connections
    mrc.add_projection("mos", "stn", 0.8)
    mrc.add_projection("mos", "cun", 0.8, to_cell_type="excitatory")
    mrc.add_projection(
        "mos", "grn_l", 0.8,
    )
    mrc.add_projection(
        "mos", "grn_r", 0.8,
    )

    mrc.add_projection("stn", "cun", 0.8, to_cell_type="inhibitory")

    mrc.add_projection(
        "cun", "mos", 0.2,
    )
    mrc.add_projection(
        "grn_l", "mos", 0.2,
    )
    mrc.add_projection(
        "grn_r", "mos", 0.2,
    )

    # ---------------------------- inputs and outputs ---------------------------- #
    mrc.add_input("mos")
    mrc.add_input("mos")
    mrc.add_input("mos")

    # specify which output comes from which region
    mrc.add_output("cun")
    mrc.add_output("grn_l", "grn_r")

    # create an RNN and visualize model weights
    rnn = RNN(
        n_units=mrc.n_units,
        input_size=mrc.n_inputs,
        output_size=mrc.n_outputs,
        connectivity=mrc.W_rec,
        input_connectivity=mrc.W_in,
        output_connectivity=mrc.W_out,
        w_in_bias=True,
        **rnn_kwargs,
    )

    return rnn, mrc


if __name__ == "__main__":
    rnn, mrc = initialize_rnn()
    plot_model_weights(rnn, **mrc.W_mtx_axes_labels)
    plt.show()
