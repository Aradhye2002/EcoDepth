from torch.optim import AdamW
from typing import List

def get_optimizer(
    model,
    args, 
    no_decay_names: List[str] = ['relative_position_bias_table', 'rpe_mlp', 'logit_scale']
):
    param_groups = []
    param_group_lookup = dict()
    for n, p in model.named_parameters():
        if p.requires_grad:
            # Determine weight decay
            if p.ndim == 1 or n.endswith(".bias") or "absolute_pos_embed" in n:
                weight_decay = 0.0
            else:
                found = False
                for no_decay_name in no_decay_names:
                    if no_decay_name in n:
                        found = True
                        break
                if found:
                    weight_decay = 0.0
                else:
                    weight_decay = args.weight_decay

            # Determine LR scale
            if "unet" in n:
                scale = 0.01
            elif "embeddings" in n:
                scale = 10.0
            else:
                scale = 1.0

            lr = scale * args.lr
            
            if (lr, weight_decay) not in param_group_lookup:
                new_param_group = {
                    "params" : [],
                    "weight_decay" : weight_decay,
                    "lr" : lr
                }
                param_group_lookup[(lr, weight_decay)] = new_param_group
                param_groups.append(new_param_group)
            
            param_group = param_group_lookup[(lr, weight_decay)]
            param_group["params"].append(p)
    
    # LR and weight_decay are already set above
    # So setting these below has no effect
    optimizer = AdamW(params=param_groups)
    return optimizer
