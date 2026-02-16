import torch.optim as optim
import torch
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.all_params = list(params)
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

        if not dist.is_initialized():
            raise RuntimeError("Distributed package not initialized")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Shard parameters
        num_params = len(self.all_params)
        # Calculate chunk_size ensuring all parameters are covered, even if not perfectly divisible
        chunk_size = (num_params + self.world_size - 1) // self.world_size
        start = self.rank * chunk_size
        end = min(start + chunk_size, num_params)
        self.local_params = self.all_params[start:end]

        # Initialize local optimizer with only the parameters in the local shard
        self.optim = self.optimizer_cls(self.local_params, **self.kwargs)

        # Initialize the base Optimizer class.
        # We pass self.local_params to the super constructor to satisfy its requirements
        # and potentially allow external tools to inspect param_groups,
        # but the actual optimization logic is delegated to self.optim.
        # We then explicitly set param_groups and state to point to the internal optimizer's.
        super().__init__(self.local_params, {})
        self.param_groups = self.optim.param_groups
        self.state = self.optim.state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step the local optimizer, updating only the parameters in the local shard
        self.optim.step()

        # Synchronize parameters across all ranks
        # Each rank broadcasts its updated shard to all other ranks
        num_all = len(self.all_params)
        chunk = (num_all + self.world_size - 1) // self.world_size

        for r in range(self.world_size):
            r_start = r * chunk
            r_end = min(r_start + chunk, num_all)

            for i in range(r_start, r_end):
                param = self.all_params[i]
                # Broadcast the updated parameter data from the rank that owns it
                dist.broadcast(param.data, src=r)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        # We must zero gradients for ALL parameters, not just local ones.
        # This is because `backward()` computes gradients for all parameters on a given rank,
        # and if non-local gradients are not zeroed, they will accumulate across steps,
        # potentially consuming memory and leading to incorrect behavior if they were ever used.
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    # Detach and zero the gradient tensor
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()
