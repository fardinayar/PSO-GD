import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class ModelHandler:
    def __init__(self, model):
        self.model = model

    @property
    def parameters(self):
        return self.model.parameters()

    def get_flat_grad(self):
        views = []
        for p in self.parameters:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def get_flat_params(self):
        return parameters_to_vector(self.parameters)

    def set_vector_to_params(self, vector):
        vector_to_parameters(vector, self.model.parameters())

    def get_loss_and_grad(self, batch_x, batch_y, loss_fn):
        y = self.model(batch_x.to(torch.float32))
        loss = loss_fn(y, batch_y)
        loss.backward()
        grad = self.get_flat_grad()
        self.model.zero_grad()
        return loss, grad

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # Get cpu or gpu device for training.
    device = 'cuda'
    print(f"Using {device} device")


    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    model = NeuralNetwork().to(device)
    mh = ModelHandler(model)
    print(mh.get_flat_grad())
    y = model(torch.ones((1, 1, 28, 28)).to(device))
    loss = (y-10)**2
    mh = ModelHandler(model)
    print(mh.get_flat_grad())
    loss.backward()
    print(mh.get_flat_grad())
    mh.zero_grad()
    print(mh.get_flat_grad())