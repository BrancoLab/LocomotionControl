import matplotlib.pyplot as plt

import sys

sys.path.append("/Users/federicoclaudi/Documents/Github/pyrnn")

from typing import Tuple

from pyrnn import CTRNN as RNN
from pyrnn.connectivity import MultiRegionConnectivity, Region


def initialize_rnn(**rnn_kwargs) -> Tuple[RNN, MultiRegionConnectivity]:
    # ------------------------------ create regions ------------------------------ #
    mos = Region(
        name="mos",
        n_units=128,
        dale_ratio=0.8,  # only excitatory
        autopses=False,
    )

    cun = Region(name="cun", n_units=64, dale_ratio=0.8, autopses=False,)

    grn = Region(name="grn", n_units=64, dale_ratio=0.8, autopses=False,)

    # ---------------------------- create connections ---------------------------- #
    mrc = MultiRegionConnectivity(mos, cun, grn)

    # create feedforward connections
    mrc.add_projection("mos", "cun", 0.1)
    mrc.add_projection("mos", "grn", 0.1)
    mrc.add_projection("cun", "mos", 0.1)
    mrc.add_projection("grn", "mos", 0.1)
    mrc.add_projection("cun", "grn", 0.1)
    mrc.add_projection("grn", "cun", 0.1)

    # ---------------------------- inputs and outputs ---------------------------- #
    mrc.add_input("mos")
    mrc.add_input("mos")
    mrc.add_input("mos")

    # specify which output comes from which region
    mrc.add_output("cun")
    mrc.add_output("grn")

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

    import numpy as np

    W = mrc.W_rec * np.random.choice([-1, 1], size=mrc.W_rec.shape)
    img = plt.imshow(W, cmap="bwr", vmin=-W.max(), vmax=W.max())

    # plot_model_weights(rnn, **mrc.W_mtx_axes_labels)
    plt.show()
