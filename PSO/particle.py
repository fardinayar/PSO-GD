import numpy as np
import torch


class Particle():
    def __init__(self, dim, device):
        '''A single particle part of the swarm.
        :param dim: int, dimension of the search space
        :param lb: (dim, ) numpy array, lower bound of the search space
        :param ub: (dim, ) numpy array, upper bound of the search space
        :param Vmax: double, Maximum allowed velocity of the particle
        '''
        self.position = None
        self.velocity = torch.zeros(dim).to(device)
        self.grad = None
        self.fitness_list = []
        self.dim = dim
        self.fitness_val = None

    @property
    def fitness(self):
        if len(self.fitness_list) > 0:
            self.fitness_val = np.mean(self.fitness_list)
        if len(self.fitness_list) > 1000:
            self.fitness_val = np.mean(self.fitness_list[-1000:])
        return self.fitness_val

    def update_pos(self):
        '''Update the particle position based on velocity.
        The position can be bounded if self.bounded is True. If position goes
        out of bound, it is set to the bound and the velocity sign is inversed.
        '''
        self.position += self.velocity

    def update_fitness_and_grad(self, fitness, grad):
        self.fitness_list.append(fitness.item())
        self.grad = grad
