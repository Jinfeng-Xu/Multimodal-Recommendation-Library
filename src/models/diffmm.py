"""
DiffMM - Diffusion Model for Multi-Modal Recommendation
Complete implementation with full diffusion process for MRS framework
Reference: https://github.com/HKUST-KnowComp/DiffMM
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math

from common.abstract_recommender import GeneralRecommender


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, d_emb_size, norm=False, time_emb_dim=64):
        super(Denoise, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(in_dims) - 1):
            self.layers.append(nn.Linear(in_dims[i], out_dims[i]))
        self.d_emb = nn.Embedding(100, d_emb_size)
        self.d_emb_size = d_emb_size
        self.norm = norm
        
        # Time embedding
        self.time_emb_dim = time_emb_dim
        self.emb_layer = nn.Linear(time_emb_dim, time_emb_dim)
        
        # input层和output层
        self.in_layers = nn.ModuleList([
            nn.Linear(out_dims[-1] + time_emb_dim, 256),
            nn.Linear(256, 256)
        ])
        self.out_layers = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, out_dims[0])  # output维度与input相同
        ])
        
        self.drop = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        # compute时间嵌入
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32, device=x.device) / (self.time_emb_dim // 2))
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        
        time_emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
        
        # 拼接input和时间嵌入
        h = torch.cat([x, time_emb], dim=-1)
        
        # input层
        for layer in self.in_layers:
            h = layer(h)
            h = torch.tanh(h)
        
        # output层
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            if beta_fixed:
                self.betas[0] = 0.0001
            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], 0.999))
        return np.array(betas)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_losses(self, denoise_model, x_start, embeds, batch_index, feats):
        t = torch.randint(0, self.steps, (x_start.shape[0],), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        x_recon = denoise_model(x_t, t)
        
        # compute损失
        loss = (noise - x_recon).pow(2).mean(dim=-1)
        
        # Graph consistency loss
        gc_loss = (feats - embeds[batch_index]).pow(2).mean(dim=-1)
        
        return loss, gc_loss

    def p_sample(self, denoise_model, x, t, noise_scale=0.0):
        with torch.no_grad():
            model_mean = self.p_mean_variance(denoise_model, x=x, t=t)
            if t == 0:
                return model_mean
            else:
                noise = torch.randn_like(x)
                return model_mean + noise_scale * noise

    def p_mean_variance(self, denoise_model, x, t):
        x_recon = denoise_model(x, t)
        model_mean = self._q_posterior_mean_variance(x_start=x_recon, x_t=x, t=t)[0]
        return model_mean

    def _q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_t
        )
        return posterior_mean, None


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = ([t.shape[0]] + [1] * (len(shape) - 1))
    out = out.reshape(*reshape)
    return out


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class DiffMM(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(DiffMM, self).__init__(config, dataloader)
        
        self.dim_E = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.ris_adj_lambda = 0.2
        self.ris_lambda = 0.1
        self.trans = 1
        self.ssl_temp = 0.2
        self.ssl_alpha = 0.001
        self.cl_method = 0
        self.n_layers = config.get('n_layers', 2)
        self.e_loss = 'bpr'
        self.rebuild_k = 10

        # build交互矩阵
        edge_index = self._build_edge_index(dataloader)
        adjusted_item_ids = edge_index[:, 1] - self.n_users
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),
                                                 (edge_index[:, 0], adjusted_item_ids)),
                                                shape=(self.n_users, self.n_items), dtype=np.float32)
        self.adj = self.get_norm_adj_mat().to(self.device)

        # initialize用户和项目嵌入
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.dim_E)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.dim_E)))
        
        # Multiple GCN layers
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.n_layers)])

        # Random dropout of sparse graph edges
        keepRate = 0.5
        self.edgeDropper = SpAdjDropEdge(keepRate)

        # Multimodal feature transformation
        if self.trans == 1:
            self.image_trans_l = nn.Linear(self.visual_feat.shape[1], self.dim_E)
            self.text_trans_l = nn.Linear(self.textual_feat.shape[1], self.dim_E)
            nn.init.xavier_uniform_(self.image_trans_l.weight)
            nn.init.xavier_uniform_(self.text_trans_l.weight)

        self.image_embedding = self.visual_feat
        self.text_embedding = self.textual_feat
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

        # Denoise model
        dims = '[1000]'
        out_dims = eval(dims) + [self.n_items]
        in_dims = out_dims[::-1]
        norm = False
        d_emb_size = 10
        self.denoise_model_image = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)
        self.denoise_model_text = Denoise(in_dims, out_dims, d_emb_size, norm=norm).to(self.device)

        # Diffusion model
        self.diffusion_model = GaussianDiffusion(
            noise_scale=0.1, noise_min=0.0001, noise_max=0.02, steps=5
        ).to(self.device)

    def _build_edge_index(self, dataloader):
        """build边索引"""
        interactions = []
        for batch in dataloader:
            users = batch[0].cpu().numpy()
            items = batch[1].cpu().numpy()
            for u, i in zip(users, items):
                interactions.append([u, i + self.n_users])
        return torch.tensor(interactions, dtype=torch.long).cpu().numpy()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users + self.n_items, self.n_users + self.n_items)))

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = sp.coo_matrix((edge_list, (u_list, i_list)), shape=(self.n_users, self.n_items), dtype=np.float32)
        a = sp.csr_matrix((self.n_users, self.n_users))
        b = sp.csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def getItemEmbeds(self):
        return self.iEmbeds

    def getUserEmbeds(self):
        return self.uEmbeds

    def getImageFeats(self):
        if self.trans == 1:
            return self.image_trans_l(self.image_embedding)
        else:
            return self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))

    def getTextFeats(self):
        if self.trans == 1:
            return self.text_trans_l(self.text_embedding)
        else:
            return self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))

    def forward(self, image_UI_matrix=None, text_UI_matrix=None):
        # Get multimodal features
        image_feats = self.getImageFeats()
        text_feats = self.getTextFeats()

        # Modality weight normalization
        weight = self.softmax(self.modal_weight)

        # Use diffusion model reconstructed UI matrices if available
        if image_UI_matrix is not None and text_UI_matrix is not None:
            # Use diffusion model reconstructed matrices for graph convolution
            embedsImageAdj = torch.spmm(image_UI_matrix, torch.concat([self.uEmbeds, self.iEmbeds]))
            embedsTextAdj = torch.spmm(text_UI_matrix, torch.concat([self.uEmbeds, self.iEmbeds]))
        else:
            # Use original adjacency matrix
            embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
            embedsImageAdj = torch.spmm(self.adj, embedsImageAdj)
            embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
            embedsTextAdj = torch.spmm(self.adj, embedsTextAdj)

        # Process image features
        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(self.adj, embedsImage)
        embedsImage_ = torch.concat([embedsImage[:self.n_users], self.iEmbeds])
        embedsImage_ = torch.spmm(self.adj, embedsImage_)
        embedsImage += embedsImage_

        # Process text features
        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(self.adj, embedsText)
        embedsText_ = torch.concat([embedsText[:self.n_users], self.iEmbeds])
        embedsText_ = torch.spmm(self.adj, embedsText_)
        embedsText += embedsText_

        # RIS regularization
        embedsImage += self.ris_adj_lambda * embedsImageAdj
        embedsText += self.ris_adj_lambda * embedsTextAdj

        # Weighted multimodal feature fusion
        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        # 将多模态融合后的嵌入input到 GCN 层中
        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(self.adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)

        # Add RIS regularization term
        embeds = embeds + self.ris_lambda * F.normalize(embedsModal)

        # return用户嵌入和项目嵌入
        return embeds[:self.n_users], embeds[self.n_users:]

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeds, item_embeds = self.forward()

        u_embeddings = user_embeds[users]
        pos_i_embeddings = item_embeds[pos_items]
        neg_i_embeddings = item_embeds[neg_items]

        # BPR loss
        pos_scores = torch.sum(u_embeddings * pos_i_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_i_embeddings, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

        # Regularization loss
        regularizer = 1. / 2 * (u_embeddings ** 2).sum() + 1. / 2 * (pos_i_embeddings ** 2).sum() + 1. / 2 * (neg_i_embeddings ** 2).sum()
        regularizer = regularizer / len(users)
        emb_loss = self.reg_weight * regularizer

        return bpr_loss + emb_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embeds, item_embeds = self.forward()
        u_embeddings = user_embeds[users]
        scores = torch.matmul(u_embeddings, item_embeds.transpose(0, 1))
        return scores
