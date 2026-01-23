import os
import random
import numpy as np
from scipy.interpolate import CubicSpline
import torch

from mrspy.sim.sim import Simulation as s


def log_loss(epoch, train_loss, test_loss, log_dir):
    log_file = os.path.join(log_dir, 'loss_log.txt')
    with open(log_file, 'a+') as f:
        f.write(f"Epoch {epoch}: Train Loss {train_loss:.8f}, Test Loss {test_loss:.8f}\n")

def random_noise_level(max_value):
    """
    return a random number from 0.002 to max_value
    """
    if max_value < 0.002:
        raise ValueError("max_value must be greater than or equal to 0.002")
    return random.uniform(0.002, max_value)


def generate_random_smooth_curves(num_points, std_factor=1.0):

    control_points_x = np.linspace(0, 1, 5)
    

    n_factor = np.random.uniform(0.5, 2.0) 
    max_val = 20 * (1 + n_factor)
    control_points_y1 = np.random.uniform(low=20, high=max_val, size=5) + np.random.normal(loc=0, scale=std_factor, size=5)
    control_points_y1 = np.clip(control_points_y1, 20, max_val)
    
    control_points_y2 = np.random.uniform(low=0, high=5, size=5) + np.random.normal(loc=0, scale=std_factor, size=5)
    control_points_y2 = np.clip(control_points_y2, 0, 5)
    
    control_points_y3 = np.random.uniform(low=0, high=5, size=5) + np.random.normal(loc=0, scale=std_factor, size=5)
    control_points_y3 = np.clip(control_points_y3, 0, 5)

    spline1 = CubicSpline(control_points_x, control_points_y1)
    spline2 = CubicSpline(control_points_x, control_points_y2)
    spline3 = CubicSpline(control_points_x, control_points_y3)

    x = np.linspace(0, 1, num_points)
    curve1 = np.clip(spline1(x), 20, max_val)
    curve2 = np.clip(spline2(x), 0, 5)
    curve3 = np.clip(spline3(x), 0, 5)
    return np.vstack([curve1, curve2, curve3])
    
def generate_random_chemical_shifts():
    a = random.uniform(-2, 6.0)
    b = random.uniform(-2, 6.0)
    c = random.uniform(-2, 6.0)
    return a, b, c

def simulation(img1, img2, img3, device, noise_level, dce_number=31):
    cfg = {
        "curve": "default",
        "device": "cuda:0",
        "chemical_shifts_cfg": "default",
        "return_type": ["gt", "no", "wei", "wei_no"],
        "wei_no": {
            "noise_level": noise_level
        },
        "no": {
            "noise_level": noise_level
        },
        "wei": {
            "average": 263,
        },
        "dtype": "float",
        "return_dict": True
    }
    cfg['device'] = device
    cfg['chemical_shifts'] = generate_random_chemical_shifts()
    cfg['curve'] = generate_random_smooth_curves(num_points=dce_number)
    sim = s(dce_number=dce_number, target_size=32, spec_len=72, cfg=cfg)
    res = sim.simulation(torch.stack([img1, img2, img3], dim=1), abs=True)
    
    return res
