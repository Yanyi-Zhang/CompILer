import torch
import torch.nn as nn
from GeM import GeM
import torch.nn.functional as F
import math


class Prompt(nn.Module):
    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
    ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.com_gem = GeM(p=3, eps=1e-6)
        self.attr_gem = GeM(p=3, eps=1e-6)
        self.obj_gem = GeM(p=3, eps=1e-6)

        self.atten = nn.MultiheadAttention(768, 1, dropout=0, batch_first=True)

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == "zero":
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == "uniform":
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

        if self.prompt_pool:
            attr_prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.attr_prompt = nn.Parameter(torch.zeros(attr_prompt_pool_shape))
            elif prompt_init == "uniform":
                self.attr_prompt = nn.Parameter(torch.randn(attr_prompt_pool_shape))
                nn.init.uniform_(self.attr_prompt, -1, 1)

        # creating attribute key
        if prompt_key:
            attr_key_shape = (pool_size, embed_dim)
            if prompt_key_init == "zero":
                self.attr_prompt_key = nn.Parameter(torch.zeros(attr_key_shape))
            elif prompt_key_init == "uniform":
                self.attr_prompt_key = nn.Parameter(torch.randn(attr_key_shape))
                nn.init.uniform_(self.attr_prompt_key, -1, 1)
        else:
            attr_prompt_mean = torch.mean(self.attr_prompt, dim=1)
            self.attr_prompt_key = attr_prompt_mean

        if self.prompt_pool:
            obj_prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == "zero":
                self.obj_prompt = nn.Parameter(torch.zeros(obj_prompt_pool_shape))
            elif prompt_init == "uniform":
                self.obj_prompt = nn.Parameter(torch.randn(obj_prompt_pool_shape))
                nn.init.uniform_(self.obj_prompt, -1, 1)

        # creating object key
        if prompt_key:
            obj_key_shape = (pool_size, embed_dim)
            if prompt_key_init == "zero":
                self.obj_prompt_key = nn.Parameter(torch.zeros(obj_key_shape))
            elif prompt_key_init == "uniform":
                self.obj_prompt_key = nn.Parameter(torch.randn(obj_key_shape))
                nn.init.uniform_(self.obj_prompt_key, -1, 1)
        else:
            obj_prompt_mean = torch.mean(self.obj_prompt, dim=1)
            self.obj_prompt_key = obj_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == "mean":
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == "max":
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == "mean_max":
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == "cls":
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            threshold = math.radians(90)
            ddl_ac = ddl_loss(self.attr_prompt, self.prompt, threshold)
            ddl_oc = ddl_loss(self.obj_prompt, self.prompt, threshold)
            ddl_ao = ddl_loss(self.attr_prompt, self.obj_prompt, threshold)
            ddl = ddl_ac + ddl_oc + ddl_ao
            out["ddl"] = ddl

            com_ortho = ortho_penalty(self.prompt.view(self.prompt.shape[0], -1))
            attr_ortho = ortho_penalty(self.attr_prompt.view(self.attr_prompt.shape[0], -1))
            obj_ortho = ortho_penalty(self.obj_prompt.view(self.obj_prompt.shape[0], -1))
            ortho = com_ortho + attr_ortho + obj_ortho
            out["ortho"] = ortho

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C
            obj_prompt_norm = self.l2_normalize(self.obj_prompt_key, dim=1)

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
            obj_similarity = torch.matmul(x_embed_norm, obj_prompt_norm.t())

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                _, obj_idx = torch.topk(obj_similarity, k=self.top_k, dim=1)
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    obj_prompt_id, obj_id_counts = torch.unique(obj_idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat(
                            [
                                prompt_id,
                                torch.full(
                                    (self.pool_size - prompt_id.shape[0],),
                                    torch.min(idx.flatten()),
                                    device=prompt_id.device,
                                ),
                            ]
                        )
                        id_counts = torch.cat(
                            [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)]
                        )
                        obj_prompt_id = torch.cat(
                            [
                                obj_prompt_id,
                                torch.full(
                                    (self.pool_size - obj_prompt_id.shape[0],),
                                    torch.min(obj_idx.flatten()),
                                    device=obj_prompt_id.device,
                                ),
                            ]
                        )
                        obj_id_counts = torch.cat(
                            [
                                obj_id_counts,
                                torch.full((self.pool_size - obj_id_counts.shape[0],), 0, device=obj_id_counts.device),
                            ]
                        )

                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k

                    _, obj_major_idx = torch.topk(obj_id_counts, k=self.top_k)  # top_k
                    obj_major_prompt_id = obj_prompt_id[obj_major_idx]  # top_k
                    obj_idx = obj_major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batched_prompt = self.com_gem(batched_prompt_raw)

            obj_batched_prompt_raw = self.obj_prompt[obj_idx]  # B, top_k, length, C
            obj_batched_prompt = self.obj_gem(obj_batched_prompt_raw)

            atten_x_embed_mean = x_embed_mean.unsqueeze(1)
            atten_x_embed_mean, _ = self.atten(atten_x_embed_mean, obj_batched_prompt, obj_batched_prompt)
            atten_x_embed_mean = atten_x_embed_mean.squeeze(1)
            atten_x_embed_norm = self.l2_normalize(atten_x_embed_mean, dim=-1)
            attr_prompt_norm = self.l2_normalize(self.attr_prompt_key, dim=1)
            attr_similarity = torch.matmul(atten_x_embed_norm, attr_prompt_norm.t())
            if prompt_mask is None:
                _, attr_idx = torch.topk(attr_similarity, k=self.top_k, dim=1)
                if self.batchwise_prompt:
                    attr_prompt_id, attr_id_counts = torch.unique(attr_idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        attr_prompt_id = torch.cat(
                            [
                                attr_prompt_id,
                                torch.full(
                                    (self.pool_size - attr_prompt_id.shape[0],),
                                    torch.min(attr_idx.flatten()),
                                    device=attr_prompt_id.device,
                                ),
                            ]
                        )
                        attr_id_counts = torch.cat(
                            [
                                attr_id_counts,
                                torch.full(
                                    (self.pool_size - attr_id_counts.shape[0],), 0, device=attr_id_counts.device
                                ),
                            ]
                        )

                    _, attr_major_idx = torch.topk(attr_id_counts, k=self.top_k)  # top_k
                    attr_major_prompt_id = attr_prompt_id[attr_major_idx]  # top_k
                    # expand to batch
                    attr_idx = attr_major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                pass

            attr_batched_prompt_raw = self.attr_prompt[attr_idx]  # B, top_k, length, C
            attr_batched_prompt = self.attr_gem(attr_batched_prompt_raw)

            out["prompt_idx"] = idx
            out["attr_prompt_idx"] = attr_idx
            out["obj_prompt_idx"] = obj_idx

            # Debugging, return sim as well
            out["prompt_norm"] = prompt_norm
            out["x_embed_norm"] = x_embed_norm
            out["similarity"] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out["selected_key"] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            attr_batched_key_norm = attr_prompt_norm[attr_idx]  # B, top_k, C
            out["attr_selected_key"] = attr_batched_key_norm
            atten_x_embed_norm = atten_x_embed_norm.unsqueeze(1)
            attr_sim = attr_batched_key_norm * atten_x_embed_norm  # B, top_k, C
            attr_reduce_sim = torch.sum(attr_sim) / x_embed.shape[0]  # Scalar

            obj_batched_key_norm = obj_prompt_norm[obj_idx]  # B, top_k, C
            out["obj_selected_key"] = obj_batched_key_norm
            obj_sim = obj_batched_key_norm * x_embed_norm  # B, top_k, C
            obj_reduce_sim = torch.sum(obj_sim) / x_embed.shape[0]  # Scalar

            out["reduce_sim"] = reduce_sim
            out["attr_reduce_sim"] = attr_reduce_sim
            out["obj_reduce_sim"] = obj_reduce_sim
        else:
            if self.prompt_init == "zero":
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == "uniform":
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out["total_prompt_len"] = batched_prompt.shape[1]
        out["attr_total_prompt_len"] = attr_batched_prompt.shape[1]
        out["obj_total_prompt_len"] = obj_batched_prompt.shape[1]
        out["prompted_embedding"] = torch.cat([batched_prompt, attr_batched_prompt, obj_batched_prompt, x_embed], dim=1)

        return out


def ddl_loss(pa, pb, threshold):
    npa = pa.view(pa.size(0), -1)
    npb = pb.view(pb.size(0), -1)

    theta = torch.acos(F.cosine_similarity(npa[:, None], npb, dim=2))
    threshold = torch.full_like(theta, threshold)

    loss = torch.sum(F.relu(threshold - theta)) * 2 / (npa.size(0) * npb.size(0))
    return loss


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).to(t.device)) ** 2).mean()
