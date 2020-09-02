import numpy as np
from collections import namedtuple
import time
import pandas as pd

from fcutils.file_io.io import save_yaml

from proj.model.symbolic import Symbolic
from proj.model.config import Config
from proj.utils import merge

class Model(Config, Symbolic):
    _M_args = ['theta', 'v', 'omega', 'L', 'R', 'm', 'd', 'm_w' , 'tau_l', 'tau_r']

    _calc_model_jacobian_state_args = ['theta', 'v', 'omega', 'L', 'R', 'm', 'd', 'm_w']
    _calc_model_jacobian_input_args = ['L', 'R', 'm', 'd', 'm_w']

    _control = namedtuple('control', 'tau_r, tau_l')
    _state = namedtuple('state', 'x, y, theta, v, omega')

    def __init__(self):
        Config.__init__(self)
        Symbolic.__init__(self)

        self._make_simbols()
        self.get_combined_dynamics_kinematics()
        # self.get_inverse_dynamics()
        self.get_jacobians()
        self.reset()

    def __repr__(self):
        return ''.join([f'{k}: -- {v}\n' for k,v in self.config_dict().items()])

    def reset(self):
        self.curr_x = self._state(0, 0, 0, 0, 0)
        self.curr_control = self._control(0, 0) # use only to keep track

        self.history = dict(
            x = [],
            y = [],
            theta = [],
            v = [],
            omega = [],
            tau_r = [],
            tau_l = [],
        )

        self._append_history() # make sure first state is included

    def _append_history(self):
        for ntuple in [self.curr_x, self.curr_control]:
            for k,v in ntuple._asdict().items():
                self.history[k].append(v)

    def save(self, trajectory):
        # Create folder
        time_stamp = time.strftime('%y%m%d_%H%M%S')
        save_fld = self.save_folder / (self.save_name + f'_{time_stamp}')
        save_fld.mkdir(exist_ok=True)

        # save config
        save_yaml(str(save_fld/'config.yml'), self.config_dict())

        # save state and control variables naes
        save_yaml(str(save_fld/'state_vars.yml'), dict(self._state(0, 0, 0, 0, 0)._asdict()))
        save_yaml(str(save_fld/'control_vars.yml'), dict(self._control(0, 0)._asdict()))

        # save trajectory
        np.save(str(save_fld/'trajectory.npy'), trajectory)

        # save history
        pd.DataFrame(self.history).to_hdf(str(save_fld/'history.h5'), key='hdf')


    def step(self, u):
        u = self._control(*np.array(u))
        self.curr_x = self._state(*self.curr_x)

        # Compute dxdt
        variables = merge(u, self.curr_x, self.mouse)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
            raise ValueError('Nans in dxdt')

        # Step
        next_x = np.array(self.curr_x) + dxdt * self.dt
        self.curr_x = self._state(*next_x)

        # Update history
        self._append_history() 
        self.curr_control = u

    def _fake_step(self, x, u):
        """
            Simulate a step fiven a state and a control
        """
        x = self._state(*x)
        u = self._control(*u)

        # Compute dxdt
        variables = merge(u, x, self.mouse)
        inputs = [variables[a] for a in self._M_args]
        dxdt = self.calc_dqdt(*inputs).ravel()

        if np.any(np.isnan(dxdt)) or np.any(np.isinf(dxdt)):
            # raise ValueError('Nans in dxdt')
            print('nans in dxdt during fake step')


        # Step
        next_x = np.array(x) + dxdt * self.dt
        return next_x

    def predict_trajectory(self, curr_x, us):
        """
            Compute the trajectory for N steps given a
            state and a (series of) control(s)
        """
        if len(us.shape) == 3:
            pred_len = us.shape[1]
            us = us.reshape((pred_len, -1))
            expand = True
        else:
            expand = False

        # get size
        pred_len = us.shape[0]

        # initialze
        x = curr_x # (3,)
        pred_xs = curr_x[np.newaxis, :]

        for t in range(pred_len):
            next_x = self._fake_step(x, us[t])
            # update
            pred_xs = np.concatenate((pred_xs, next_x[np.newaxis, :]), axis=0)
            x = next_x

        if expand:
            pred_xs = pred_xs[np.newaxis, :, :]
            # pred_xs = np.transpose(pred_xs, (1, 0, 2))
        return pred_xs

    def calc_gradient(self, xs, us, wrt='x'):
        """
            Compute the models gradient wrt state or control
        """

        # prep some variables
        theta = xs[:, 2]
        v = xs[:, 3]
        omega = xs[:, 4]

        L = self.mouse['L']
        R = self.mouse['R']
        m = self.mouse['m']
        m_w = self.mouse['m_w']
        d = self.mouse['d']

        # reshapshapee
        (_, state_size) = xs.shape
        (pred_len, input_size) = us.shape

        res = []
        if wrt == 'x':
            f = np.zeros((pred_len, state_size, state_size))
            for i in range(pred_len):
                f[i, :, :] = self.calc_model_jacobian_state(theta[i], v[i], omega[i], L, R, m, d, m_w)
    
                return f * self.dt + np.eye(state_size)
        else:
            f = np.zeros((pred_len, state_size, input_size))
            f0 = self.calc_model_jacobian_input(L, R, m, d, m_w) # no need to iterate because const.
            for i in range(pred_len):
                f[i, :, :] = f0

            return f * self.dt

