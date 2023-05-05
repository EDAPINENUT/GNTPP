import numpy as np
import torch
# from sklearn.metrics import classification_report
def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

def numerical_expectation(prob, log_weights, t_max: float = 40.0, resolution: int = 100):
    device = log_weights.device
    with torch.no_grad():
        time_step = t_max / resolution
        x_axis = torch.linspace(1e-7, t_max, resolution).to(device)
        batch_size, seq_len, event_type_num, mix_number = log_weights.shape
        x_axis = x_axis[None,None,None,:,None].expand(batch_size, seq_len, event_type_num, -1, mix_number)
        heights = prob(x_axis)
        weights = log_weights.exp()
        component_expectation = (x_axis * heights * time_step).sum(dim=-2) * weights
        expectation = component_expectation.sum(dim=-1)
    return expectation

def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()
    
def calculate_metrics(label, predict):
    sample_num = label.shape[0]
    predict_label = np.where(predict>0.5, 1, 0)
    acc = (predict_label == label.astype(int).flatten()).sum()/sample_num
    report = classification_report(label, predict_label, output_dict=True)
    return acc, report

def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def encode_onehot(y):
    classes = set(y)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    y_onehot = np.array(list(map(classes_dict.get, y)),
                                dtype=np.int32)
    return y_onehot

class NonNeg:
    """
    Constrains the weights to be non-negative.
    """
    def __call__(self, module):
        w = module.weight.data
        module.weight.data = w.gt(0).float().mul(w)

class Diagonal:
    """
    Constrains the weights to be diagonal.
    """
    def __init__(self, event_type_num: int, embed_dim: int):
        self.n = event_type_num
        self.e = embed_dim      # dimension for each event type

    def __call__(self, module):
        w = module.weight.data
        mask = torch.zeros_like(w)
        for i in range(self.n):
            s, e = i * self.e, (i + 1) * self.e
            mask[s:e, s:e] = 1
        module.weight.data = w.mul(mask)

def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]