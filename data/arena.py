from dataclasses import dataclass
from myterial import salmon, blue_light, red_light, green, indigo, teal

'''
    Define the position of ROIs in the arena
'''

@dataclass
class ROI:
    name: str
    x_0: int  # position of top left corner in cm
    y_0: int
    x_1: int  # position of top right corner in cm
    y_1: int
    g_0: float # start in global coord
    g_1: float # end in global coord
    color: str



T1 = ROI('T1', 8, 24, 0, 20, 0.055, .235, salmon)
T2 = ROI('T2', 8, 32, 33, 52, 0.27, .41, blue_light)
T3 = ROI('T3', 24, 40, 0, 20, 0.45, .625, red_light)
T4 = ROI('T4', 24, 40, 0, 20, 0.70, .85, indigo)

S1 = ROI('S1', 32, 40, 20, 45, 0.635, 0.695, green)
S2 = ROI('S2', 32, 40, 20, 45, 0.87, 0.98, teal)

ROIs = [T1, T2, T3, T4, S1, S2]
ROIs_dict = dict(
    T1 = T1, 
    T2 = T2, 
    T3 = T3, 
    T4 = T4,
    S1 = S1, 
    S2 = S2,
)


