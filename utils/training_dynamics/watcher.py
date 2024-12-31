import torch

from collections import OrderedDict, defaultdict

class MagnitudeTracker(object):
    def __init__(self, ):
        super().__init__()

        self.val_dict = OrderedDict()
        self.cur_dict = OrderedDict()
        self.ctr_dict = defaultdict(lambda : 0)

        self.param_grad_d = OrderedDict()
        self.param_data_d = OrderedDict()
    
    def make_entry(self, name):
        self.val_dict[name] = None
        self.cur_dict[name] = None
    
    def track_output(self, module, name):
        self.make_entry(name)

        def hook(mod, arg, out):
            with torch.no_grad():
                if isinstance(out, (list, tuple)):
                    for i, val in enumerate(out):
                        key = f"{name}.{i}"
                        if not torch.is_tensor(val):
                            continue
                        if key not in self.val_dict:
                            self.make_entry(key)
                        self.add(key, torch.linalg.vector_norm(val.float(), dim=list(range(1, val.ndim))).mean())
                else:
                    self.add(name, torch.linalg.vector_norm(out.float(), dim=list(range(1, out.ndim))).mean())
        module.register_forward_hook(hook)
    
    # def watch_weight(self, module, name):
    #     self.make_entry(name)

    #     if hasattr(module, "weight"):
    #         def hook(mod, arg, out):
    #             self.add(name, mod.weight)
    #         module.register_forward_hook(hook)

        # def hook(g):
        #     self.add(name, g)
        #     return g

        # tensor.register_hook(hook)

    def track_weight(self, parameter, name):
        self.make_entry(name)
        self.param_data_d[name] = parameter

    def track_gradient(self, parameter, name):
        self.make_entry(name)
        self.param_grad_d[name] = parameter

    def manual_trigger_weight(self):
        with torch.no_grad():
            for n, p in self.param_data_d.items():
                self.add(n, torch.linalg.vector_norm(p.data))
    
    def manual_trigger_gradient(self):
        with torch.no_grad():
            for n, p in self.param_grad_d.items():
                if p.grad is not None:
                    self.add(n, torch.linalg.vector_norm(p.grad))

    def add(self, name, value):
        with torch.no_grad():
            # DDP?
            if self.val_dict[name] is None:
                self.val_dict[name] = value.detach().float().clone()
            else:
                self.val_dict[name].add_(value)
            
            self.cur_dict[name] = value.detach().clone()
            self.ctr_dict[name] += 1
    
    def clear(self):
        for k in self.val_dict:
            self.val_dict[k] = None
        for k in self.ctr_dict:
            self.ctr_dict[k] = 0
    
    def as_dict(self):

        ret = OrderedDict()

        for k, v in self.val_dict.items():
            if v is not None:
                ret[k] = v / self.ctr_dict[k]

        cur = OrderedDict(self.cur_dict)
        
        return ret, cur

class TensorTracker(object):
    def __init__(self, ):
        super().__init__()

        self.val_dict = OrderedDict()
        self.ctr_dict = defaultdict(lambda : 0)

    def add(self, name, value):
        with torch.no_grad():
            # DDP?
            if name not in self.val_dict:
                self.val_dict[name] = value.detach().clone()
            else:
                self.val_dict[name].add_(value)
            
            self.ctr_dict[name] += 1
            
    def clear(self):
        for k in self.val_dict:
            if torch.is_tensor(self.val_dict[k]):
                self.val_dict[k].zero_()
            else:
                del self.val_dict[k]
        for k in self.ctr_dict:
            self.ctr_dict[k] = 0
    
    def as_dict(self):

        ret = OrderedDict()

        for k, v in self.val_dict.items():
            if v is not None:
                ret[k] = v / self.ctr_dict[k]

        return ret
