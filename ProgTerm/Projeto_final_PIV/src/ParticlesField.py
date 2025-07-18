from .piv_domain import PIVDomain
import numpy as np

class ParticlesField():
    def __init__(self, domain:PIVDomain):
        self.domain = domain
        self.positions = self.create_particles()
    def create_particles(self):
        x = np.random.randint(0,self.domain.width,self.domain.num_particles)
        y = np.random.randint(0,self.domain.height,self.domain.num_particles)
        print(x)
        return np.vstack((x, y)).T

    def create_image(self):
        image = np.zeros((self.domain.height, self.domain.width), dtype=np.uint8)
        for x,y in self.positions:
            image[y,x] = 255
        return image


