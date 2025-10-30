# MPS 兼容版本的 NativeScaler
# 修复了 CUDA 特定的 GradScaler

import torch

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, device_type='cuda'):
        self.device_type = device_type
        if device_type == 'cuda' and torch.cuda.is_available():
            try:
                self._scaler = torch.amp.GradScaler('cuda')
            except:
                # 兼容旧版本
                self._scaler = torch.cuda.amp.GradScaler()
        else:
            # MPS 或 CPU 不使用 scaler
            self._scaler = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        if self._scaler is not None:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
        else:
            # 不使用 scaler 的情况
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = get_grad_norm_(parameters)
                optimizer.step()
            else:
                norm = None
        return norm

    def state_dict(self):
        if self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


