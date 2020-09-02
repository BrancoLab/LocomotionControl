from logging import getLogger

import numpy as np

def fit_angle_in_range(angles, min_angle=-np.pi, max_angle=np.pi, is_deg=True):
    """ Check angle range and correct the range
    it assumes that the angles are passed ind degrees
    
    Args:
        angle (numpy.ndarray): in radians
        min_angle (float): maximum of range in radians, default -pi
        max_angle (float): minimum of range in radians, default pi
    Returns: 
        fitted_angle (numpy.ndarray): range angle in radians
    """
    if max_angle < min_angle:
        raise ValueError("max angle must be greater than min angle")
    if (max_angle - min_angle) < 2.0 * np.pi:
        raise ValueError("difference between max_angle \
                          and min_angle must be greater than 2.0 * pi")
    if is_deg:
        output = np.radians(angles)
    else:
        output = np.array(angles)

    output_shape = output.shape

    output = output.flatten()
    output -= min_angle
    output %= 2 * np.pi
    output += 2 * np.pi
    output %= 2 * np.pi
    output += min_angle

    output = np.minimum(max_angle, np.maximum(min_angle, output))
    output =  output.reshape(output_shape)

    # if is_deg:
    #     output = np.degrees(output)
    return output

def calc_cost(pred_xs, input_sample, g_xs,
              state_cost_fn, input_cost_fn, terminal_state_cost_fn):
    """ calculate the cost 

    Args:
        pred_xs (numpy.ndarray): predicted state trajectory, 
            shape(pop_size, pred_len+1, state_size)
        input_sample (numpy.ndarray): inputs samples trajectory,
            shape(pop_size, pred_len+1, input_size)
        g_xs (numpy.ndarray): goal state trajectory,
            shape(pop_size, pred_len+1, state_size)
        state_cost_fn (function): state cost fucntion
        input_cost_fn (function): input cost fucntion
        terminal_state_cost_fn (function): terminal state cost fucntion
    Returns:
        cost (numpy.ndarray): cost of the input sample, shape(pop_size, )
    """
    # state cost
    state_cost = 0.
    if state_cost_fn is not None:
        state_pred_par_cost = state_cost_fn(pred_xs[:, 1:-1, :], g_xs[:, 1:-1, :])
        state_cost = np.sum(np.sum(state_pred_par_cost, axis=-1), axis=-1)

    # terminal cost
    terminal_state_cost = 0.
    if terminal_state_cost_fn is not None:
        terminal_state_par_cost = terminal_state_cost_fn(pred_xs[:, -1, :],
                                                        g_xs[:, -1, :])
        terminal_state_cost = np.sum(terminal_state_par_cost, axis=-1)
    
    # act cost
    act_cost = 0.
    if input_cost_fn is not None:
        act_pred_par_cost = input_cost_fn(input_sample)
        act_cost = np.sum(np.sum(act_pred_par_cost, axis=-1), axis=-1)

    return state_cost + terminal_state_cost + act_cost