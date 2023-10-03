import numpy as np
import torch

from PSO.pso import PSO
from PSO.particle import Particle
from ModelHandler.model_handler import ModelHandler
from torch.nn.init import xavier_uniform_

class Optimizer:
    def __init__(self, model, n_particles=8, Vmax=None, inertial_weight=0.0001, self_conf=0.001, swarm_conf=0.005, norm=True):
        self.model = ModelHandler(model)
        self.iter = 0
        self.device = next(model.parameters()).device
        self.n_particles = n_particles
        self.dim = len(self.model.get_flat_params())
        self.pso = PSO(dim=self.dim, Vmax=Vmax,inertial_weight=inertial_weight, self_conf=self_conf, swarm_conf=swarm_conf, norm=norm)
        self.los_history = []
        #self.flag = int(swarm_conf)
    def step(self, batch_x, batch_y, loss_fn):
        self.iter += 1
        if self.iter == 1:
            for i in range(self.n_particles):
                particle = Particle(self.dim, self.device)
                #np.random.seed(i)
                weights = torch.tensor(np.random.normal(0,0.1,self.dim)).to(torch.float32).to(self.device)
                particle.position = weights
                self.pso.particles.append(particle)

        '''if self.flag:
            self.pso.C2 = (self.iter/10000) ** 3 * 0.5'''

        '''if self.iter > 4000:
            self.pso.w = 0.001

        if self.iter > 5000:
            self.pso.norm = False'''

        for particle in self.pso.particles:
            self.model.set_vector_to_params(particle.position)
            particle.update_fitness_and_grad(*self.model.get_loss_and_grad(batch_x, batch_y, loss_fn))

        best_pos, best_val = self.pso.step()
        self.model.set_vector_to_params(best_pos)
        self.los_history.append(best_val)
        return best_val





