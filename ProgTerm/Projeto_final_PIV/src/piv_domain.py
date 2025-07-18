class PIVDomain:
    def __init__(self, width, height, part_density=0.01):
        self.width = width
        self.height = height
        self.part_density = part_density
        self.num_particles = int(width * height * part_density)

