
from .ode_func import CNF, ODEfunc, ODEdiff
import torch.nn as nn

def build_flow(embed_size, layer_num, time_length=0.1, train_T=True, solver='dopri5', use_adjoint=True,
               atol=1e-5, rtol=1e-5):
    def build_cnf():
        diffeq = ODEdiff(
            embed_size=embed_size,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=time_length,
            train_T=train_T,
            solver=solver,
            use_adjoint=use_adjoint,
            atol=atol,
            rtol=rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(layer_num)]
    model = SequentialFlow(chain)
    return model


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, dt, history_embedding, logpt=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        dt = dt.reshape(-1, 1)
        history_embedding = history_embedding.reshape(-1, history_embedding.shape[-1])
        logpt = logpt.reshape(-1, logpt.shape[-1]) if logpt is not None else None
        if logpt is None:
            for i in inds:
                dt = self.chain[i](
                    dt, 
                    history_embedding=history_embedding,
                    logpt=logpt, 
                    integration_times=integration_times, 
                    reverse=reverse
                )
            return dt
        else:
            for i in inds:
                dt, logpt = self.chain[i](
                    dt, 
                    history_embedding=history_embedding, 
                    logpt=logpt, 
                    integration_times=integration_times, 
                    reverse=reverse
                )
            return dt, logpt