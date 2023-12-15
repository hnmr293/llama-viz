def fix_seed(seed: int, deterministic=True):
    # ref.
    # https://qiita.com/north_redwing/items/1e153139125d37829d2d
    # https://take-tech-engineer.com/pytorch-randam-seed-fix/
    
    import random
    random.seed(seed)
    
    import numpy
    numpy.random.seed(seed)
    
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms = True
