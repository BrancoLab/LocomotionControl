import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from proj.rnn.tasks.control_task import ControlTask
from proj.rnn.backend.models.basic import Basic as RNN
