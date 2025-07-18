import numpy as np
class VelocityField():
    def __init__(self, height,width):
        self.height = height
        self.width = width
        self.u_field = None
        self.v_field = None

    def generate_velocity_field(self, u_velocity = 1.0, v_velocity = 1.0):
        self.u_field = np.full((self.height, self.width), u_velocity)
        self.v_field = np.full((self.height, self.width), v_velocity)
        print(self.u_field)