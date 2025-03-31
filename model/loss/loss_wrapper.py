import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    return tensors_gather

class DistributedLossWrapper(torch.nn.Module):
    def __init__(self, loss, dim, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        # self.loss = DDP(loss, **kwargs) if has_parameters else loss
        if has_parameters:
            print('Loss {} has parameters and is wrapped by *DDP*'.format(loss))
            self.loss = DDP(loss, **kwargs)
        else:
            print('Loss {} has *NO* parameters and is *NOT* wrapped by *DDP*'.format(loss))
            self.loss = loss
        self.dim = dim
        
    def forward(self, embeddings, labels, *args, **kwargs):
        # embeddings, labels = all_gather(embeddings, labels, dim=self.dim)
        all_embeddings = torch.cat(GatherLayer.apply(embeddings), dim=self.dim)
        all_labels = torch.cat(all_gather(labels), dim=self.dim)
        # print("gather embeddings shape: {}".format(all_embeddings.size()))
        # print("gather labels shape: {}".format(all_labels.size()))
        return self.loss(all_embeddings, all_labels, *args, **kwargs)

class DistributedLossWrapperWithLabelType(torch.nn.Module):
    def __init__(self, loss, dim, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        # self.loss = DDP(loss, **kwargs) if has_parameters else loss
        if has_parameters:
            print('Loss {} has parameters and is wrapped by *DDP*'.format(loss))
            self.loss = DDP(loss, **kwargs)
        else:
            print('Loss {} has *NO* parameters and is *NOT* wrapped by *DDP*'.format(loss))
            self.loss = loss
        self.dim = dim
        
    def forward(self, embeddings, labels, labels_type, *args, **kwargs):
        # embeddings, labels = all_gather(embeddings, labels, dim=self.dim)
        all_embeddings = torch.cat(GatherLayer.apply(embeddings), dim=self.dim)
        all_labels = torch.cat(all_gather(labels), dim=self.dim)
        all_labels_type = torch.cat(all_gather(labels_type), dim=self.dim)
        # print("gather embeddings shape: {}".format(all_embeddings.size()))
        # print("gather labels shape: {}".format(all_labels.size()))
        return self.loss(all_embeddings, all_labels, all_labels_type, *args, **kwargs)

class DistributedLossWrapperWithLabelTypeWithLabelOrigin(torch.nn.Module):
    def __init__(self, loss, dim, **kwargs):
        super().__init__()
        has_parameters = len([p for p in loss.parameters()]) > 0
        # self.loss = DDP(loss, **kwargs) if has_parameters else loss
        if has_parameters:
            print('Loss {} has parameters and is wrapped by *DDP*'.format(loss))
            self.loss = DDP(loss, **kwargs)
        else:
            print('Loss {} has *NO* parameters and is *NOT* wrapped by *DDP*'.format(loss))
            self.loss = loss
        self.dim = dim
        
    def forward(self, embeddings, labels, labels_type, labels_origin, *args, **kwargs):
        # embeddings, labels = all_gather(embeddings, labels, dim=self.dim)
        all_embeddings = torch.cat(GatherLayer.apply(embeddings), dim=self.dim)
        all_labels = torch.cat(all_gather(labels), dim=self.dim)
        all_labels_type = torch.cat(all_gather(labels_type), dim=self.dim)
        all_labels_origin = torch.cat(all_gather(labels_origin), dim=self.dim)
        # print("gather embeddings shape: {}".format(all_embeddings.size()))
        # print("gather labels shape: {}".format(all_labels.size()))
        return self.loss(all_embeddings, all_labels, all_labels_type, all_labels_origin, *args, **kwargs)
