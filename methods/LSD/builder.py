# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from LSD.models import PatchTST_LSD_MAE, PatchTST_LSD
import LSD.models.backbone_ablation as backbone_ablation
from LSD.models.backbone_abla_models import MultiheadAttention_ROPE, LlamaRMSNorm

# MAE
import mup
from mup import MuReadout

class BYOL_LSD(nn.Module):
    """
    Build a BYOL model
    by wnz 2024/2/23
    """

    def __init__(self, base_encoder, input_length=10000, dim=256, mlp_dim=4096):
        super(BYOL_LSD, self).__init__()
        # build encoders
        if base_encoder == PatchTST_LSD.PatchTST_samll:
            base_backbone = PatchTST_LSD.PatchTST_samll(input_length=input_length, head_type='cls')
            momentum_backbone = PatchTST_LSD.PatchTST_samll(input_length=input_length, head_type='cls')
            output_dim = 192

            self.base_encoder = nn.Sequential(
                base_backbone,  # B, D
                self._build_mlp(2, output_dim, mlp_dim, dim)
            )

            self.momentum_encoder = nn.Sequential(
                momentum_backbone,  # B, D
                self._build_mlp(2, output_dim, mlp_dim, dim)
            )
        else:
            # todo: support more backbones
            raise NotImplementedError()
        
        self.predictor = self._build_mlp(3, dim, mlp_dim, dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient


    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim):
        """
        Build BYOL's projector and predictor MLPs.
        BYOL's mlps don't have bn in the last layer.
        """
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                #mlp.append(nn.LayerNorm(dim2))
                mlp.append(nn.ReLU(inplace=True))

        return nn.Sequential(*mlp)

    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=1, p=2)
        y = F.normalize(y, dim=1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    def sim_loss(self, x, y):
        x = F.normalize(x, dim=1, p=2)
        y = F.normalize(y, dim=1, p=2)
        similarity = torch.einsum("nc,nc->n", [x, y])
        loss = -similarity.mean()
        return loss
      
    @torch.no_grad()
    def compute_keys(self, k, m):
        keys = []
        with torch.no_grad():
            self._update_momentum_encoder(m)
            for i in range(len(k)):
                keys.append(self.momentum_encoder(k[i]))
        return keys
      
    def forward(self, x1, x2, m, sequential=False):
        if sequential:
            q = self.predictor(self.base_encoder(x1))
            loss = self.loss_fn(q, x2[0])
            for i in range(1, len(x2)):
                loss += self.loss_fn(q, x2[i])
            loss = loss.mean()/len(x2)
            return loss
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(m)

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        loss = (self.loss_fn(q1, k2) + self.loss_fn(q2, k1)).mean()

        return loss

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class MAE_LSD(nn.Module):
    """
    Build a MAE model
    by lhl 2024/3/7
    """

    def __init__(self, 
                 base_encoder, 
                 base_decoder, 
                 mask_ratio, 
                 mask_way,
                 loss_type,
                 norm_pix_loss,
                 input_length=10000, 
                 patch_len=50, 
                 encoder_size=None,
                 decoder_size=None,
                 channel_way='dependent', 
                 args=None):
        super(MAE_LSD, self).__init__()
        
        self.patch_len = patch_len
        if channel_way.startswith('independent'):
            c_in = 1
        else:
            c_in = 3
        self.channel_way = channel_way
        self.c_in = c_in
        
        # build encoders
        if base_encoder == backbone_ablation.Encoder_baseline_llama:
            self.base_encoder = backbone_ablation.Encoder_baseline_llama(
                encoder_size, input_length=10000, c_in=c_in, args=args
            )
        elif base_encoder == backbone_ablation.Encoder_llama_bias:
            self.base_encoder = backbone_ablation.Encoder_llama_bias(
                encoder_size, input_length=10000, c_in=c_in, args=args
            )
        else:
            raise NotImplementedError()

        # build decoders
        if base_decoder == backbone_ablation.Decoder_baseline_llama:
            self.base_decoder = backbone_ablation.Decoder_baseline_llama(
                encoder_dim=self.base_encoder.backbone.d_model,
                decoder_size=decoder_size,
                input_length=input_length,
                c_in=c_in,
                args=args
            )
        elif base_decoder == backbone_ablation.Decoder_llama_bias:
            self.base_decoder = backbone_ablation.Decoder_llama_bias(
                encoder_dim=self.base_encoder.backbone.d_model,
                decoder_size=decoder_size,
                input_length=input_length,
                c_in=c_in,
                args=args
            )
        else:
            # todo: support more backbones
            raise NotImplementedError()
        
        if base_encoder != PatchTST_LSD_MAE.PatchTSTEncoder_base_vit:
            self.base_encoder.backbone.set_mask(
                mask_ratio=mask_ratio,
                mask_way=mask_way
            )
        else:
            print("model PatchTSTEncoder_base_vit do not support `set mask`")
            
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss
        self.args = args
    
    def _init_weights(self, module, readout_zero_init=False, query_zero_init=False):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            ### muP: swap constant std normal init with normal_ from `mup.init`.
            ### Because `_init_weights` is called in `__init__`, before `infshape` is set,
            ### we need to manually call `self.apply(self._init_weights)` after calling
            ### `set_base_shape(model, base)`
            if isinstance(module, MuReadout) and readout_zero_init:
                module.weight.data.zero_()
            else:
                if hasattr(module.weight, 'infshape'):
                    # mup.init.normal_(module.weight, std=self.args.init_std)
                    mup.init.xavier_uniform_(module.weight)
                else:
                    # module.weight.data.normal_(mean=0.0, std=self.args.init_std)
                    nn.init.xavier_uniform_(module.weight)
            ### End muP
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias'):
                module.bias.data.zero_()
        ### muP
        if isinstance(module, MultiheadAttention_ROPE):
            if query_zero_init:
                module.q_proj.weight.data[:] = 0
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, L)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_len

        l = imgs.shape[2] // p # patch_num
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], l, p))
        x = torch.einsum('nclp->nlpc', x)
        x = x.reshape(shape=(imgs.shape[0], l, p * imgs.shape[1]))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, p*3)
        imgs: (N, 3, L)
        """
        p = self.patch_len
        l = x.shape[1]
        if self.channel_way.startswith('independent'):
            x = x.reshape(shape=(x.shape[0]//3, l, p, 3))
            x = torch.einsum('nlpc->nclp', x)
            imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        else:
            x = x.reshape(shape=(x.shape[0], l, p, 3))
            x = torch.einsum('nlpc->nclp', x)
            imgs = x.reshape(shape=(x.shape[0], 3 , l * p))
        return imgs
    
    def forward_loss(self, x, pred, mask):
        """
        x: [bs x n_vars x seq_num]
        pred: [bs x patch_num x nvars * patch_len]
        mask: [bs x patch_num], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x) # target: [bs x patch_num x nvars * patch_len]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if self.loss_type == 'l1':
            loss = (pred - target).abs()
        elif self.loss_type == 'l2':
            loss = (pred - target) ** 2
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        else:
            raise NotImplementedError()

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, x_feature=None):
        '''
        x: [bs x nvars x seq_len]
        '''
        if self.channel_way.startswith('independent'):
            x = x.reshape(x.size(0)*x.size(1),1,x.size(2))
            if self.channel_way == 'independent_shuffle':
                x = x[torch.randperm(x.size(0))]
        latent, mask, ids_restore = self.base_encoder(x) # [bs x nvars x seq_len] -> [bs x (kept_num + 1) x encoder_dim], mask=ids_restore: [bs x patch_num]
        pred = self.base_decoder(latent, ids_restore) # [bs x (kept_num + 1) x encoder_dim] -> [bs x patch_num x nvars * patch_len]
        if x_feature is not None:
            x = x_feature
        loss = self.forward_loss(x, pred, mask)
        # return loss, pred, mask
        return loss


# if __name__ == "__main__":
#     model = MAE_LSD(                
#                 base_encoder=PatchTST_LSD.PatchTSTEncoder_base_vit, 
#                 base_decoder=PatchTST_LSD.PatchTSTDecoder_common_vit,
#                 mask_ratio = 0.75
#                 )
#     from torchviz import make_dot
#     input = torch.randn(8, 3, 10000)
#     # loss, pred, mask = model(input)
#     output = model(input)
#     make_dot(output, params=dict(model.named_parameters())).render("model_graph_patchtst", format="png")

    # print(loss, pred.shape, mask.shape)