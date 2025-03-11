import torch

class Merger:
    def __init__(self,method,expert_weight):
        self.method = method
        self.expert_weight=expert_weight
    
    def merge(self, experts, mapping,debugging=False):
        if 'uniform' in self.method:
            return self.uniform(experts,mapping,debugging)
        else:
            raise NotImplementedError(self.method)
    
    def uniform(self, experts, mapping,debugging=False):
        if isinstance(mapping, list):
            mapping = torch.tensor(mapping)
        elif isinstance(mapping, torch.Tensor):
            pass
        else:
            raise NotImplementedError(type(mapping))
        num_experts = len(mapping)
        if debugging:
            w = getattr(experts[-1], 'w1').weight
            print(w)
        # 1. GET & SET merged expert
        for q in range(num_experts):
            # print('q')
            indices = torch.where(mapping == q)[0]
            if len(indices)==0:
                continue
            # assert indices[0] == q
            for weight_name in self.expert_weight:
                w = getattr(experts[q], weight_name).weight
                for idx in indices:
                    if idx == q:
                        # w = getattr(experts[idx], weight_name).weight
                        continue
                    else:
                        w = w + getattr(experts[idx], weight_name).weight
                w=w/(len(indices))
                w=torch.nn.Parameter(w)
                if w.device != getattr(experts[q], weight_name).weight.device:
                    w.to(getattr(experts[q], weight_name).weight.device)
                getattr(experts[q], weight_name).weight=w

        # 2. UNSET other expert
        for q in range(num_experts):
            indices = torch.where(mapping == q)[0]
            if len(indices)!=0:
                continue
            for weight_name in self.expert_weight:
                w = getattr(experts[q], weight_name).weight
                w = w*0
                w = torch.nn.Parameter(w)
                getattr(experts[q], weight_name).weight=w
        
        torch.cuda.empty_cache()
        if debugging:
            w = getattr(experts[-1], 'w1').weight
            print(w)
    
    