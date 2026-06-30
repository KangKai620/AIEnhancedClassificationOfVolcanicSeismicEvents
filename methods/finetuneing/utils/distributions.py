import torch
import numpy as np

class GaussianMixture():
    def __init__(self,num=2,weights=torch.Tensor([0.5,0.5]),mus=torch.Tensor([0,1]),stds=torch.Tensor([1,1])):
        self.num = num
        self.weights = weights
        self.mus = mus
        self.stds = stds
    
    def set_params(self,params):
        '''params (Tensor): shape (*,L), params[:,:L/3]=mixture weights, params[:,L/3:2L/3] mus, params[:,2L/3:] sigmas'''
        assert params.shape[-1] % 3 == 0
        self.num = params.shape[-1]//3
        self.weights = params[:,:self.num]
        # self.weights = torch.nn.functional.gumbel_softmax(params[:,:self.num], tau=1, dim=-1)
        self.mus = params[:,self.num:2*self.num]
        self.stds = params[:,2*self.num:]

    def prob(self,obs,eps=1.0e-8):
        if self.weights.dim() > obs.dim(): # reshape obs so that it can be added with self.mus
            diff = self.weights.dim() - obs.dim()
            obs = obs.reshape(*obs.shape,*(1,)*diff)
        # likelihood
        ll = torch.exp(- 0.5* ((obs - self.mus)/(self.stds+eps)) ** 2) / (self.stds + eps)/np.sqrt(2.*np.pi)
        res = torch.sum((self.weights * ll), dim = -1)
        return res

    def log_prob(self, obs, eps=1.0e-8):
        if self.weights.dim() > obs.dim(): # reshape obs so that it can be added with self.mus
            diff = self.weights.dim() - obs.dim()
            obs = obs.reshape(*obs.shape,*(1,)*diff)

        # log likelihood
        component_ll = - 0.5* ((obs - self.mus)/(self.stds+eps)) ** 2 - torch.log(self.stds + eps) - 0.5*np.log(np.pi*2.)
        res = torch.logsumexp(torch.log(self.weights + eps) + component_ll, dim=-1)
        # res = torch.logsumexp(component_ll, dim=-1)

        return res
    
    def mean(self):
        return torch.sum(self.weights * self.mus, dim=-1, keepdim=True)

    def maxprob_val(self,startv=-5,endv=10,stepv=0.1,deg=2):
        '''using grid search between [-5,10] is enough'''
        grid = torch.arange(startv, endv, stepv).reshape(-1,1,1).to(self.mus.device)
        probs = self.prob(grid)
        grid = grid[torch.argmax(probs,dim=0)].ravel()
        step = 0.5
        for i in range(deg-1):
            tmp = []
            step *= 0.1
            for j in range(len(grid)):
                tmp.append(torch.linspace(grid[j]-step,grid[j]+step,11).reshape(-1,1,1).to(self.mus.device))
            grid = torch.cat(tmp,dim=1)
            probs = self.prob(grid)
            grid = grid[torch.argmax(probs,dim=0),np.arange(self.mus.shape[0])].ravel()
        return grid.reshape(-1,1)

    # need modification
    # def sample(self):
    #     """Draw samples from a MoG."""
    #     categorical = Categorical(pi)
    #     pis = categorical.sample().unsqueeze(1)
    #     sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
    #     # Gathering from the n Gaussian Distribution based on sampled indices
    #     sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
    #     return sample
    
    # def generate_samples(self, pi, sigma, mu, n_samples=None):
    #     if n_samples is None:
    #         n_samples = self.hparams.n_samples
    #     samples = []
    #     softmax_pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
    #     assert (
    #         softmax_pi < 0
    #     ).sum().item() == 0, "pi parameter should not have negative"
    #     for _ in range(n_samples):
    #         samples.append(self.sample(softmax_pi, sigma, mu))
    #     samples = torch.cat(samples, dim=1)
    #     return samples

    # def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
    #     # Sample using n_samples and take average
    #     samples = self.generate_samples(pi, sigma, mu, n_samples)
    #     if self.hparams.central_tendency == "mean":
    #         y_hat = torch.mean(samples, dim=-1)
    #     elif self.hparams.central_tendency == "median":
    #         y_hat = torch.median(samples, dim=-1).values
    #     return y_hat
