import numpy as np
import torch

from .particle import Particle
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

class PSO():
    def __init__(self, dim, Vmax=None, inertial_weight=0.9, self_conf=0.9, swarm_conf=0.1, norm=True):
        '''Wrapper for running particle swarm optimization
        :param cost_fn: function which returns the cost to be optimized for one input
        :param n_particles: numebr of aprticles to use in PSO
        :param dim: int, dimension of the search space
        :param lb: (dim, ) numpy array, lower bound of the search space
        :param ub: (dim, ) numpy array, upper bound of the search space
        :param Vmax: double, Maximum allowed velocity of the particle. Default inf
        :param inertial_weight: float, hyperparameter determining the effect of the velocity on next iteration. Default 0.5
        :param self_conf: float, C1 hyperparameter self-confidence. Default 0.8
        :param swarm_conf: float, C2 hyperparameter swarm-confidence. Default 0.9
        :param n_iter: int, number of iterations of PSO to run. Default = 500
        :param target: float, target minimum value to achieve. Default is None
        :param target_error: float, stopping criterion if given target value
        :param bounded: bool, if True, the positions are kept within lb and ub. If false, lb and ub only used for particle initialization
        '''
        self.particles = []
        self.gbest_pos = np.random.normal(dim)
        self.gbest_value = np.inf
        self.Vmax, self.w, self.C1, self.C2 = Vmax, inertial_weight, self_conf, swarm_conf
        self.best_particle = None
        self.costs = None
        self.norm = norm

    def update_best(self):
        '''Update the global and local best for the set of particles
        '''
        self.costs = [particle.fitness for particle in self.particles]
        self.costs = np.array(self.costs)
        mincost_id = self.costs.argmin()
        mincost = self.costs[mincost_id]
        if mincost < self.gbest_value:
            self.gbest_value = mincost
            self.gbest_pos = self.particles[mincost_id].position.clone()

    def move_part(self, p: Particle):
        ''' Update the velocity of single particle and perform movement
        :param p: Particle object to move
        '''
        norm = torch.norm(p.grad) if self.norm else 1
        new_velocity = (self.w * p.velocity) - (self.C1 * np.random.random() * p.grad / norm) + (self.C2 * np.random.random() * (self.gbest_pos - p.position))
        new_velocity = torch.clip(new_velocity, -0.1,0.1) #khodam
        p.velocity = new_velocity
        p.update_pos()

    def move_particles(self):
        '''Perform velcoity update on all particles
        '''
        for paticle in self.particles:
            self.move_part(paticle)

    def step(self):
        '''Run the PSO algorithm
        '''
        self.update_best()
        self.move_particles()
        return self.gbest_pos, self.gbest_value




if __name__ == '__main__':
    def f(x):
        return (x[0] - 1) ** 2 + (x[1] - 4) ** 2, np.array([2 * (x[0] - 1), 2 * (x[1] - 4)])
    pso = PSO(f, 10, 2)
    for i in range(100):
        [particle.update_fitness_and_grad(*f(particle.position)) for particle in pso.particles]
        print(pso.step())