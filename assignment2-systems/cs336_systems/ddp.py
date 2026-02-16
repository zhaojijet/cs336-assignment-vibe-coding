import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # Broadcast initial weights from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for backward pass
        self._grad_handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Use a default value for param in the lambda to capture it correctly
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param_module):
        def hook(param):
            # Asynchronously reduce gradients
            if param.grad is None:
                return

            # We want to sum gradients across all ranks.
            # dist.all_reduce is in-place on the gradient tensor.
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._grad_handles.append((handle, param.grad))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # Wait for all async reductions to complete
        for handle, grad in self._grad_handles:
            handle.wait()
            # Normalize gradients by world size
            # Use .data to avoid "leaf Variable that requires grad" error
            grad.data.div_(dist.get_world_size())
        self._grad_handles.clear()


class DDPBucketed(nn.Module):
    def __init__(self, module, bucket_size_mb):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        # Broadcast initial weights
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Group parameters into buckets
        # We iterate in reverse order of parameters() as a heuristic for backward order
        all_params = [p for p in self.module.parameters() if p.requires_grad]
        all_params.reverse()

        self.buckets = []
        current_bucket_params = []
        current_bucket_size = 0

        for param in all_params:
            param_size = param.numel() * param.element_size()
            if (
                current_bucket_size + param_size > self.bucket_size_bytes
                and current_bucket_params
            ):
                # Finish current bucket
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0

            current_bucket_params.append(param)
            current_bucket_size += param_size

        if current_bucket_params:
            self._create_bucket(current_bucket_params)

        # Param to bucket mapping
        self.param_to_bucket_idx = {}
        for i, bucket in enumerate(self.buckets):
            for param in bucket["params"]:
                # Use id(param) or param itself as key? param is hashable?
                # nn.Parameter is hashable (by object id).
                self.param_to_bucket_idx[param] = i

        # Register hooks
        for param in all_params:
            param.register_post_accumulate_grad_hook(self._make_hook())

    def _create_bucket(self, params):
        # Flatten parameters into a buffer
        # We don't create the buffer tensor yet?
        # We need a buffer for GRADIENTS.
        # Size = sum of numel.
        total_numel = sum(p.numel() for p in params)
        # We assume all params have same dtype.
        dtype = params[0].dtype if params else torch.float32

        bucket = {
            "params": params,
            "total_numel": total_numel,
            "dtype": dtype,
            "buffer": None,  # To be allocated on first backward? Or now?
            # Creating buffer now allows reusing it.
            # But we need it on correct device.
            "device": params[0].device,
            "handle": None,
            "count": 0,
            "offsets": {},  # param -> (start, end)
        }

        # Calculate offsets
        offset = 0
        for p in params:
            numel = p.numel()
            bucket["offsets"][p] = (offset, offset + numel)
            offset += numel

        # Allocate buffer on device
        bucket["buffer"] = torch.zeros(
            total_numel, dtype=dtype, device=bucket["device"]
        )

        self.buckets.append(bucket)

    def _make_hook(self):
        def hook(param):
            if param.grad is None:
                return

            idx = self.param_to_bucket_idx.get(param)
            if idx is None:
                return

            bucket = self.buckets[idx]
            bucket["count"] += 1

            # Copy grad to buffer
            start, end = bucket["offsets"][param]
            # Flatten grad and copy
            bucket["buffer"][start:end].copy_(param.grad.view(-1))

            if bucket["count"] == len(bucket["params"]):
                # All params in bucket ready, trigger all_reduce
                # Asynchronously reduce
                handle = dist.all_reduce(
                    bucket["buffer"], op=dist.ReduceOp.SUM, async_op=True
                )
                bucket["handle"] = handle

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def on_train_batch_start(self):
        # Reset counters
        for bucket in self.buckets:
            bucket["count"] = 0
            bucket["handle"] = None

    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            # Check for incomplete buckets that need flushing
            if bucket["handle"] is None and bucket["count"] > 0:
                bucket["handle"] = dist.all_reduce(
                    bucket["buffer"], op=dist.ReduceOp.SUM, async_op=True
                )

            # If bucket started reduction (was full or flushed), wait for it
            if bucket["handle"] is not None:
                bucket["handle"].wait()

                # Scatter back to params
                # Divide by world size
                # Use .data to avoid "leaf Variable that requires grad" error
                bucket["buffer"].data.div_(dist.get_world_size())

                for p in bucket["params"]:
                    start, end = bucket["offsets"][p]
                    # Only copy back if the parameter has a gradient
                    if p.grad is not None:
                        # We are modifying p.grad in place.
                        # Reshape buffer slice to p.grad shape
                        p.grad.data.view(-1).copy_(bucket["buffer"][start:end])

        # Reset for next batch (bucket counters are reset in on_train_batch_start,
        # but handles need to be cleared/checked)
        # Actually handles are replaced next time.
        # But good to clear refs.
        for bucket in self.buckets:
            bucket["handle"] = None
            bucket["count"] = 0
