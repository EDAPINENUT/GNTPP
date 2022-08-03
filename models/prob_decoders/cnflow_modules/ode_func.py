import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal
from models.embeddings.time import *
from .ode_units import *

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super().__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, t, history_embedding, logpt=None, integration_times=None, reverse=False):
        if logpt is None:
            _logpt = torch.zeros(*t.shape[:-1], 1).to(t)
        else:
            _logpt = logpt

        states = (t, _logpt, history_embedding)
        atol = self.atol
        rtol = self.rtol

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(t), self.sqrt_end_time * self.sqrt_end_time]
                ).to(t)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(t)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # print(integration_times)

        # Refresh the odefunc statistics,
        # and prepare the graph.
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(t),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(t),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpt is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class ODEdiff(nn.Module):
    def __init__(self, 
                 embed_size) -> None:
        super().__init__()

        self.embed_size = embed_size
        self.network = ODEmlp((embed_size, ), (embed_size, ))
        self.network_t = nn.Linear(1, self.embed_size)
        self.network_h = nn.Linear(self.embed_size, self.embed_size)
        self.out_network = ConcatSquashLinear(self.embed_size, 1, dim_c=0)
        self.activation = nn.GELU()
    
    def forward(self, t, dt, history_embedding):
        dt_emb = self.network_t(dt)
        h_emb = self.network_h(history_embedding)
        emb = dt_emb + h_emb 
        return self.activation(self.out_network(t, emb))


def divergence_approx(f, y, e=None):
    # e_dzdx = torch.tensor(0).to(f)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True, retain_graph=True)[0].contiguous()
    e_dzdx_e = e_dzdx.mul(e)
    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    # assert approx_tr_dzdx.requires_grad, \
    #     "(failed to add embedding) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
    #     % (
    #     f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    
    return approx_tr_dzdx


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.divergence_fn = divergence_approx
        self.register_buffer("_num_evals", torch.tensor(0.))
    
    def forward(self, t, states):
        y = states[0]
        t = torch.ones(*y.shape[:-1], 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)
            
        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)
        
        with torch.enable_grad():
            assert len(states) == 3  # conditional CNF: x, logpx, history_embedding
            history_embedding = states[-1]
            dy = self.diffeq(t, y, history_embedding)
            divergence = self.divergence_fn(dy, y, e=self._e).unsqueeze(-1)       
        return dy, -divergence, torch.zeros_like(history_embedding)
        
    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

