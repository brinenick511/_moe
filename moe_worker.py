import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,'

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import json
from tqdm import tqdm
import dei_utils as Dei
import matplotlib.pyplot as plt
import seaborn as sns
import os

from grouping.merge_methods import Merger

class MoEWorker:
    def __init__(self,name):
        home_dir = os.path.expanduser("~")
        self.name = name
        self.model_path = None
        self.json_file_path = None
        self.shard_path = None
        self.num_layer = None
        self.num_expert = None
        self.expert_path = None
        self.expert_weight = None
        self.router_path = None
        self.hidden_size = None
        
        if 'Mixtral' in name:
            if 'Instruct' in name:
                self.model_path = home_dir+'/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
                self.json_file_path = "/new_data/yanghq/models/mistralai/Mixtral-8x7B-Instruct-v0.1/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/mistralai/Mixtral-8x7B-Instruct-v0.1/{}'
            else:
                self.model_path = home_dir+'/models/mistralai/Mixtral-8x7B-v0.1'
                self.json_file_path = "/new_data/yanghq/models/mistralai/Mixtral-8x7B-v0.1/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/mistralai/Mixtral-8x7B-v0.1/{}'
            self.num_layer = 32
            self.num_expert = 8
            self.expert_path = 'model.layers.{}.block_sparse_moe.experts.{}.{}.weight'
            self.expert_weight = ['w1','w2','w3',]
            self.router_path = 'model.layers.{}.block_sparse_moe.gate.weight'
            self.hidden_size = 4096
            
        elif 'Qwen' in name:
            self.model_path = home_dir+'/models/Qwen/Qwen1.5-MoE-A2.7B'
            self.json_file_path = "/new_data/yanghq/models/Qwen/Qwen1.5-MoE-A2.7B/model.safetensors.index.json"
            self.shard_path = home_dir+'/models/Qwen/Qwen1.5-MoE-A2.7B/{}'
            self.num_layer = 24
            self.num_expert = 60
            self.expert_path = 'model.layers.{}.mlp.experts.{}.{}.weight'
            self.expert_weight = ['up_proj','gate_proj','down_proj',]
            self.router_path = 'model.layers.{}.mlp.gate.weight'
            self.hidden_size = 2048
            
        elif 'DeepSeek' in name:
            if 'Chat' in name:
                self.model_path = home_dir+'/models/deepseek-ai/DeepSeek-V2-Lite-Chat'
                self.json_file_path = "/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite-Chat/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/deepseek-ai/DeepSeek-V2-Lite-Chat/{}'
            else:
                self.model_path = home_dir+'/models/deepseek-ai/DeepSeek-V2-Lite'
                self.json_file_path = "/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/deepseek-ai/DeepSeek-V2-Lite/{}'
            self.num_layer = 27
            self.num_expert = 64
            self.expert_path = 'model.layers.{}.mlp.experts.{}.{}.weight'
            self.expert_weight = ['up_proj','gate_proj','down_proj',]
            self.router_path = 'model.layers.{}.mlp.gate.weight'
            self.hidden_size = 2048
            
        elif 'Moonlight' in name:
            if 'Instruct' in name:
                self.model_path = home_dir+'/models/moonshotai/Moonlight-16B-A3B-Instruct'
                self.json_file_path = "/new_data/yanghq/models/moonshotai/Moonlight-16B-A3B-Instruct/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/moonshotai/Moonlight-16B-A3B-Instruct/{}'
            else:
                self.model_path = home_dir+'/models/moonshotai/Moonlight-16B-A3B'
                self.json_file_path = "/new_data/yanghq/models/moonshotai/Moonlight-16B-A3B/model.safetensors.index.json"
                self.shard_path = home_dir+'/models/moonshotai/Moonlight-16B-A3B/{}'
            self.num_layer = 27
            self.num_expert = 64
            self.expert_path = 'model.layers.{}.mlp.experts.{}.{}.weight'
            self.expert_weight = ['up_proj','gate_proj','down_proj',]
            self.router_path = 'model.layers.{}.mlp.gate.weight'
            self.hidden_size = 2048
            
        else:
            raise NotImplementedError(name)
        tmp_l = self.model_path.split('/')
        self.model_name = f'{tmp_l[-2]}/{tmp_l[-1]}'
    
    def merge(self,model=None,mappings=None,method='uniform'):
        self.merger = Merger(method,self.expert_weight)
        if model:
            self.model = model
        for layer in range(self.num_layer):
            # print(f'merging layer # {layer:02d}')
            if layer==0 and ('DeepSeek' in self.name or 'Moonlight' in self.name):
                continue
            experts=model.model.layers[layer]
            if 'Mixtral' in self.name:
                experts=experts.block_sparse_moe.experts
            else:
                experts=experts.mlp.experts
            self.merger.merge(experts, mappings[layer],False)
    
    def get_expert(self,layer,flatten=True):
        # json_file_path = "/new_data/yanghq/models/mistralai/Mixtral-8x7B-Instruct-v0.1/model.safetensors.index.json"
        json_file_path = self.json_file_path
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            data=data["weight_map"]
        shard_list=[]
        for i in range(self.num_expert):
            for j in self.expert_weight:
                weight_idx = self.expert_path.format(layer,i,j)
                shard_path = data[weight_idx]
                if shard_path not in shard_list:
                    shard_list.append(shard_path)
        weight_dict={}
        for shard_path in shard_list:
            # shard_path = fhome_dir+'/models/mistralai/Mixtral-8x7B-v0.1/{shard_path}'
            shard_path = self.shard_path.format(shard_path)
            shard_weights = load_file(shard_path)
            for i in range(self.num_expert):
                for j in self.expert_weight:
                    weight_idx = self.expert_path.format(layer,i,j)
                    if weight_idx in shard_weights.keys():
                        weight_dict.update({weight_idx:shard_weights[weight_idx]})
        # TODO: reshape and return
        assert len(weight_dict) == 3*(self.num_expert)
        weight_list=[]
        for w in weight_dict.values():
            if w.shape[0] == self.hidden_size:
                w=w.transpose(0,1)
            weight_list.append(w)
        weights=torch.stack(weight_list)
        if flatten:
            weights = weights.reshape([self.num_expert,-1])
        else:
            weights = weights.reshape([self.num_expert,3,self.hidden_size,-1])
        return weights
    
    def get_router(self,layer):
        # json_file_path = "/new_data/yanghq/models/mistralai/Mixtral-8x7B-Instruct-v0.1/model.safetensors.index.json"
        json_file_path = self.json_file_path
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        data=data["weight_map"]
        # weight_idx = f'model.layers.{layer}.block_sparse_moe.gate.weight'
        weight_idx = self.router_path.format(layer)
        shard_path = data[weight_idx]
        # shard_path = fhome_dir+'/models/mistralai/Mixtral-8x7B-v0.1/{shard_path}'
        shard_path = self.shard_path.format(shard_path)
        shard_weights = load_file(shard_path)
        weights = shard_weights[weight_idx]
        return weights


def get_intra_dist():
    # model_name_list = ['Mixtral-8x7B-v0.1','Qwen1.5-MoE-A2.7B','DeepSeek-V2-Lite','Moonlight-16B-A3B',]
    model_name_list = [
        'Mixtral-8x7B-v0.1','Mixtral-8x7B-Instruct-v0.1','Qwen1.5-MoE-A2.7B',
        'DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B','Moonlight-16B-A3B-Instruct',]
    
    print('[GET] exp_cos, rou_cos, exp_mse, rou_mse')
    for model_name in model_name_list:
        print(f'\n[MODEL_NAME] {model_name}')
        loader = MoEWorker(model_name)
        exp_cos_sim = torch.zeros(loader.num_layer, loader.num_expert, loader.num_expert)-100.0
        rou_cos_sim = torch.zeros(loader.num_layer, loader.num_expert, loader.num_expert)-100.0
        exp_mse_sim = torch.zeros(loader.num_layer, loader.num_expert, loader.num_expert)-100.0
        rou_mse_sim = torch.zeros(loader.num_layer, loader.num_expert, loader.num_expert)-100.0
        
        for layer in tqdm(range(loader.num_layer)):
            if layer==0 and ('DeepSeek' in model_name or 'Moonlight' in model_name):
                continue
            # if layer not in [1,2,3,]:
            #     continue
            torch.cuda.empty_cache()
            experts = loader.get_expert(layer,True).to(torch.double).to('cuda')
            router = loader.get_router(layer).to(torch.double).to('cuda')
            for i in range(loader.num_expert):
                for j in range(i+1):
                    exp_cos_sim[layer, i, j] = F.cosine_similarity(experts[i].unsqueeze(0), experts[j].unsqueeze(0))
                    exp_cos_sim[layer, j, i] = exp_cos_sim[layer, i, j]
                    rou_cos_sim[layer, i, j] = F.cosine_similarity(router[i].unsqueeze(0), router[j].unsqueeze(0))
                    rou_cos_sim[layer, j, i] = rou_cos_sim[layer, i, j]
                    diff = experts[i] - experts[j]
                    exp_mse_sim[layer, i, j] = torch.sum(diff ** 2) / diff.numel()
                    exp_mse_sim[layer, j, i] = exp_mse_sim[layer, i, j]
                    diff = router[i] - router[j]
                    rou_mse_sim[layer, i, j] = torch.sum(diff ** 2) / diff.numel()
                    rou_mse_sim[layer, j, i] = rou_mse_sim[layer, i, j]
        Dei.save(exp_cos_sim, f'dist/exp_cos/{model_name}')
        Dei.save(rou_cos_sim, f'dist/rou_cos/{model_name}')
        Dei.save(exp_mse_sim, f'dist/exp_mse/{model_name}')
        Dei.save(rou_mse_sim, f'dist/rou_mse/{model_name}')
        
        del loader, exp_cos_sim,rou_cos_sim
    
    print('[FINISH] exp_cos, rou_cos, exp_mse, rou_mse')



def get_inter_dist():
    model_name_list = [
        'Mixtral-8x7B-v0.1','Mixtral-8x7B-Instruct-v0.1','Qwen1.5-MoE-A2.7B',
        'DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B','Moonlight-16B-A3B-Instruct',]
    model_name_list = ['Mixtral-8x7B-v0.1']
    print('[GET] inter_exp_cos')
    alt = Dei.Alternator(1)
    for model_name in model_name_list:
        print(f'\n[MODEL_NAME] {model_name}')
        loader = MoEWorker(model_name)
        exp_cos_sim = torch.zeros(loader.num_layer, loader.num_expert, loader.num_expert)-100.0
        experts_lower=None
        experts_upper=None
        for layer in tqdm(range(loader.num_layer - 1)):
            if layer==0 and ('DeepSeek' in model_name or 'Moonlight' in model_name):
                continue
            # if layer not in [1,2,3,]:
            #     continue
            torch.cuda.empty_cache()
            if experts_upper is None:
                experts_lower = loader.get_expert(layer,True).to(torch.bfloat16).to(f'cuda:{alt.next()}')
            else:
                experts_lower = experts_upper
            experts_upper = loader.get_expert(layer+1,True).to(torch.bfloat16).to(f'cuda:{alt.next()}')
            for i in range(loader.num_expert):
                for j in range(loader.num_expert):
                    exp_cos_sim[layer, i, j] = F.cosine_similarity(experts_lower[i].unsqueeze(0), experts_upper[j].unsqueeze(0))
                    # exp_cos_sim[layer, j, i] = F.cosine_similarity(experts_lower[j].unsqueeze(0), experts_upper[i].unsqueeze(0))
        Dei.save(exp_cos_sim, f'dist/inter_exp_cos/{model_name}')
        
        del loader, exp_cos_sim
    
    print('[FINISH] inter_exp_cos')

def draw_intra_dist():
    def plot_heatmap(tensor,title=None, save_path=None):
        plt.figure(figsize=(4, 3))
        sns.heatmap(tensor.numpy(), cmap='viridis', 
                    vmin=0, vmax=1,
                    # , fmt='.2f', annot=True
                    )
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        plt.close()
    model_name_list = [
        'Mixtral-8x7B-v0.1','Mixtral-8x7B-Instruct-v0.1','Qwen1.5-MoE-A2.7B',
        'DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B','Moonlight-16B-A3B-Instruct',]
    model_name_list = [
        'DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B','Moonlight-16B-A3B-Instruct',
        'Mixtral-8x7B-v0.1','Mixtral-8x7B-Instruct-v0.1','Qwen1.5-MoE-A2.7B',
        ]
    # model_name_list = ['DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B']
    home_dir = os.path.expanduser("~")
    for model_name in model_name_list:
        print(f'\n[MODEL_NAME] {model_name}')
        dirs = f'{home_dir}/data/intra_sim_heatmap/{model_name}'
        os.makedirs(dirs, exist_ok=True)
        loader = MoEWorker(model_name)
        data = Dei.load(f'dist/exp_cos/{model_name}')
        for layer in tqdm(range(loader.num_layer)):
            title=f'intra_cos/{model_name}/{layer:02d}'
            save_path=f'{home_dir}/data/intra_sim_heatmap/{model_name}/{layer:02d}.png'
            plot_heatmap(data[layer],title,save_path)

def test_merge(model_name='Mixtral-8x7B-v0.1'):
    import random
    print(f'------')
    print(f'{model_name}')
    print(f'------')
    worker = MoEWorker(model_name)
    model_path = worker.model_path
    print(model_path)
    print(f'------')
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if 'Lite' in worker.model_path:
        # from ..models.modeling_deepseek import DeepseekV2ForCausalLM
        from models.modeling_deepseek import DeepseekV2ForCausalLM
        print('___using deepseek___')
        model = DeepseekV2ForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()

    def gen(n=30):
        text = "who are you?"
        # text = 'the famous joke is'
        # text = ["who are you?",'the famous joke is','I think the cutest pet in the world is cat, but many people dont agree with me']
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt",padding=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=n,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            
            )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    gen()
    num_expert = 8*(worker.num_expert//8)
    mapping = list(range(num_expert//2))
    for _ in range(num_expert//2):
        mapping.append(random.randint(0,num_expert//2-1))
    mapping+=list(range(num_expert,worker.num_expert))
    print(mapping)
    mappings = []
    for _ in range(worker.num_layer):
        mappings.append(mapping)
    worker.merge(model,mappings)
    gen()
    del model

if __name__ == "__main__":
    print('Hello SMoE!')
    # get_intra_dist()
    # get_inter_dist()
    # draw_intra_dist()
    model_name_list = [
        'DeepSeek-V2-Lite','DeepSeek-V2-Lite-Chat','Moonlight-16B-A3B','Moonlight-16B-A3B-Instruct',
        'Mixtral-8x7B-v0.1','Mixtral-8x7B-Instruct-v0.1','Qwen1.5-MoE-A2.7B',
        ]
    model_name_list = [
        'DeepSeek-V2-Lite','Moonlight-16B-A3B',
        'Mixtral-8x7B-v0.1','Qwen1.5-MoE-A2.7B',
        ]
    model_name_list = [
        'DeepSeek-V2-Lite',
    ]
    for model_name in model_name_list:
        test_merge(model_name)
    
