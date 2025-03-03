import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat ,pack
import sys
#
from extensions.pointops.functions import pointops  
#from pytorch3d.ops.points_normals import estimate_pointcloud_normals
#from timm.layers import trunc_normal_  
from timm.models.layers import trunc_normal_ #老版本 pt虚拟环境
from utils.logger import *
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1,dcd,ChamferDistanceL2,DGCNN_loss,Point_NN,calc_cd_like_InfoV2,calc_cd_like_hyperV2

from apes import APESClsBackbone, Embedding

from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        center = pointops.fps(xyz, self.num_group)
        idx = pointops.knn(center, xyz, self.group_size)[0]
        neighborhood = pointops.index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, feat_dim):
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )

    def forward(self, x):
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        return feature_global


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)

        pcd = self.mlp1(z).reshape(bs, -1, 3)  #  B M C(3)

        return pcd


class ManifoldnessConstraint(nn.Module):
    """
    The Normal Consistency Constraint
    """
    def __init__(self, support=8, neighborhood_size=32):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.support = support
        self.neighborhood_size = neighborhood_size

    def forward(self, xyz):

        normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)

        idx = pointops.knn(xyz, xyz, self.support)[0]
        neighborhood = pointops.index_points(normals, idx)

        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        penalty = 1 - cos_similarity
        penalty = penalty.std(-1)
        penalty = penalty.mean(-1)
        return penalty


@MODELS.register_module()
class UnPlantPC(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        # define parameters
        self.config = config
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.mask_ratio = config.mask_ratio
        self.feat_dim = config.feat_dim
        self.n_points = config.n_points
        self.nbr_ratio = config.nbr_ratio

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)

        # init weights
        self.apply(self._init_weights)
        # init loss
        self._get_lossfnc_and_weights(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.manifold_constraint = ManifoldnessConstraint(support=config.support, neighborhood_size=config.neighborhood_size)
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.manifold_weight = config.manifold_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        # group points
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat  = self.encoder(rebuild_points.view(B, -1, 3))

        # complete shape generation
        pred = self.generator(feat).contiguous()

        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        # normal consistency constraint
        #manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss# + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, latent_recon_loss


    def forward(self, partial, n_points=None, record=False):
        # group points
        B, _, _ = partial.shape
        feat = self.encoder(partial)
        pred_c = self.generator(feat).contiguous()


        return pred
# ##****原始 baseline版本


def UEPS(xyz, points, npoint, alpha):
    device = points.device
    B, C, N = points.shape
    
    # Initialize centroids and distance
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float('inf'), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        # Note: xyz is now (B, 3, N), so we need to adjust the indexing
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
        
        # Efficient Euclidean distance calculation using broadcasting
        dist1 = torch.sum((xyz - centroid) ** 2, dim=1)
        
        # Efficient cosine distance calculation
        dot_product = torch.sum(xyz * centroid, dim=1)
        xyz_norm = torch.norm(xyz, p=2, dim=1)
        centroid_norm = torch.norm(centroid, p=2, dim=1).squeeze(-1)
        norm_product = xyz_norm * centroid_norm.unsqueeze(-1)
        norm_product = torch.clamp(norm_product, min=1e-10)
        cosine_similarity = dot_product / norm_product
        dist2 = 1 - cosine_similarity
        
        # Combine distances using alpha
        dist = alpha * dist1 + (1 - alpha) * dist2
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    # Index points using the sampled indices
    view_shape = list(centroids.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(centroids.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    
    # Adjust for new dimensions
    sampled_xyz = xyz[batch_indices, :, centroids].permute(0, 2, 1).contiguous()   # (B, 3, M)
    sampled_points = points[batch_indices, :, centroids].permute(0, 2, 1).contiguous()   # (B, C, M)
    
    return sampled_xyz, sampled_points
    
def fps_downsample(coor, x, num_group):
    xyz = coor.transpose(1, 2).contiguous() # b, n, 3
    #fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
    fps_idx = furthest_point_sample(xyz, num_group)#furthest_point_sample
    combined_x = torch.cat([coor, x], dim=1)

    new_combined_x = (
        pointnet2_utils.gather_operation(
            combined_x, fps_idx
        )
    )

    new_coor = new_combined_x[:, :3]
    new_x = new_combined_x[:, 3:]

    return new_coor, new_x
# 


@MODELS.register_module()
class UnPlantPC(nn.Module):
 
    def __init__(self, config, **kwargs):
        super().__init__()
        # define parameters
        self.config = config
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.mask_ratio = config.mask_ratio
        self.feat_dim = config.feat_dim
        self.n_points = config.n_points
        self.nbr_ratio = config.nbr_ratio

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)

        # init weights
        self.apply(self._init_weights)
        # init loss
        self._get_lossfnc_and_weights(config)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.manifold_constraint = ManifoldnessConstraint(support=config.support, neighborhood_size=config.neighborhood_size)
        self.shape_matching_weight = config.shape_matching_weight
        self.shape_recon_weight = config.shape_recon_weight
        self.latent_weight = config.latent_weight
        self.manifold_weight = config.manifold_weight

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        # group points
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        feat  = self.encoder(rebuild_points.view(B, -1, 3))

        # complete shape generation
        pred = self.generator(feat).contiguous()

        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        # normal consistency constraint
        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()
        #print(manifold_penalty)

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty


# #     def forward(self, partial, n_points=None, record=False):
# #         # group points
# #         B, _, _ = partial.shape
# #         feat = self.encoder(partial)
# #         pred = self.generator(feat).contiguous()
# #         return pred

# class Encoder(nn.Module):
#     def __init__(self, feat_dim):
#         """
#         PCN based encoder
#         """
#         super().__init__()

#         self.first_conv = nn.Sequential(
#             nn.Conv1d(3, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(128, 256, 1)
#         )
#         self.second_conv = nn.Sequential(
#             nn.Conv1d(512, 512, 1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(512, feat_dim, 1)
#         )

#     def forward(self, x):
#         bs, n, _ = x.shape
#         feature = self.first_conv(x.transpose(2, 1))  # B 256 n
#         feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
#         feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
#         feature = self.second_conv(feature)  # B 1024 n
#         feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
#         ##* 1108 he
#        # feature_global = feature_global.div(feature_global.norm(dim=-1, keepdim=True))  
#         return feature_global
class Encoder_fps(nn.Module):
    def __init__(self, feat_dim, n_knn=20):
        """
        PCN based encoder
        """
        super().__init__()
        self.embedding = Embedding()
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.transformer_2 = vTransformer(128, dim=128, n_knn=n_knn)
        self.transformer_3 = vTransformer(128, dim=256, n_knn=n_knn)
        self.conv = nn.Conv1d(128, 1024, 1)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        #fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        fps_idx = furthest_point_sample(xyz, num_group)#furthest_point_sample
        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def forward(self, x):
        self.res_link_list = []
        coor=x
       # print('x',x.shape)
        x = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        x=self.transformer_1(x.contiguous(),coor.contiguous())
        #x = self.n2p_attention(x)  # (B, 128, 2048) -> (B, 128, 2048)
        self.res_link_list.append(self.conv(x).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        #print(coor.shape+x.shape)
        coor, x = fps_downsample(coor, x,1024) # (B, 128, 2048) -> (B, 128, 1024)
        x=self.transformer_2(x.contiguous(),coor.contiguous())
        # x = self.n2p_attention1(x)  # (B, 128, 1024) -> (B, 128, 1024)
        self.res_link_list.append(self.conv1(x).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)

        coor, x = fps_downsample(coor, x,256)
        x=self.transformer_3(x.contiguous(),coor.contiguous())
        # x = self.n2p_attention2(x)  # (B, 128, 512) -> (B, 128, 512)
        self.res_link_list.append(self.conv2(x).max(dim=-1)[0])  # (B, 128, 512) -> (B, 1024, 512) -> (B, 1024)
        x, ps = pack(self.res_link_list, 'B *') 
     #   print(x.shape)
        #  print(self.res_link_list[2].shape) # (B, 3072)
        return x
class Encoder_ueps(nn.Module):
    def __init__(self, feat_dim, n_knn=20):
        """
        PCN based encoder
        """
        super().__init__()
        self.embedding = Embedding()
        self.transformer_1 = vTransformer(128, dim=64, n_knn=n_knn)
        self.transformer_2 = vTransformer(128, dim=128, n_knn=n_knn)
        self.transformer_3 = vTransformer(128, dim=256, n_knn=n_knn)
        self.conv = nn.Conv1d(128, 1024, 1)
        self.conv1 = nn.Conv1d(128, 1024, 1)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.alpha=0.5
       # self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        self.res_link_list = []
        coor=x
        #print(x.shape)
        x = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        x=self.transformer_1(x.contiguous(),coor.contiguous())
        #x = self.n2p_attention(x)  # (B, 128, 2048) -> (B, 128, 2048)
        self.res_link_list.append(self.conv(x).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        #print(coor.shape+x.shape)
        coor, x = UEPS(coor, x,1024,self.alpha) # (B, 128, 2048) -> (B, 128, 1024)
        x=self.transformer_2(x,coor)
        # x = self.n2p_attention1(x)  # (B, 128, 1024) -> (B, 128, 1024)
        self.res_link_list.append(self.conv1(x).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)

        coor, x = UEPS(coor, x,512,self.alpha)
        x=self.transformer_3(x,coor)
        # x = self.n2p_attention2(x)  # (B, 128, 512) -> (B, 128, 512)
        self.res_link_list.append(self.conv2(x).max(dim=-1)[0])  # (B, 128, 512) -> (B, 1024, 512) -> (B, 1024)
        x, ps = pack(self.res_link_list, 'B *') 
     #   print(x.shape)
        #  print(self.res_link_list[2].shape) # (B, 3072)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)
       # print(z.shape)
        pcd = self.mlp1(z).reshape(bs, -1, 3)  #  B M C(3)

        return pcd

class Refiner(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp2 = nn.Sequential(
            nn.Linear(4096*3, 2048*3),
            nn.ReLU(inplace=True),
            nn.Linear(2048*3, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)
        z_flattened = z.view(z.size(0), -1)
        pred1=self.mlp2(z_flattened)
        pred1 = pred1.view(pred1.size(0), 2048, 3)
        # print(pred1.shape)
        pcd = pred1.reshape(bs, -1, 3) 
        #pcd = self.mlp2(z).reshape(bs, -1, 3)  #  B M C(3)

        return pcd




    def calc_cd_like_InfoV2(p1, p2):


        dist1, dist2, idx1, idx2 = ChamferDistanceL1(p1, p2)
        dist1 = torch.clamp(dist1, min=1e-9)
        dist2 = torch.clamp(dist2, min=1e-9)
        d1 = torch.sqrt(dist1)
        d2 = torch.sqrt(dist2)

        distances1 = - torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
        distances2 = - torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

        return (torch.sum(distances1) + torch.sum(distances2)) / 2
    
      
    
@MODELS.register_module()
class UnPlantPC(nn.Module):
    def __init__(self, config=None, **kwargs):
        if config is None:
            self.config = {}
        else:
            self.config = config
    # def __init__(self, config, **kwargs):
        super().__init__()
        # define parameters
        self.config = config
        # self.num_group = config.num_group
        # self.group_size = config.group_size
        # self.mask_ratio = config.mask_ratio
        # self.feat_dim = config.feat_dim
        # self.n_points = config.n_points
        # self.nbr_ratio = config.nbr_ratio
        self.num_group = 64
        self.group_size = 32
        self.mask_ratio = [24, 32, 8]
        self.feat_dim = 3072
        ##       self.feat_dim =3072
        #self.feat_dim =3072
        self.n_points =1024
        self.nbr_ratio = 2.0
        self.support = 24
        self.neighborhood_size =32
        self.in_channels=3

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(self.feat_dim)
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)
        self.encoder_apes3 =APESClsBackbone('local',256,32)
        self.encoder_ueps = Encoder_ueps(self.feat_dim, n_knn=20)
        #self.encoder_fps = Encoder_ueps(self.feat_dim, n_knn=20)
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()
       # self.shape_criterion = calc_cd_like_InfoV2()     #self.shape_criterion = ChamferDistanceL1()
        #self.shape_criterion =  calc_cd_like_InfoV2()
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        self.shape_criterion_info = calc_cd_like_InfoV2()
        self.shape_criterion_hyper =  lambda p1, p2: calc_cd_like_hyperV2(p1, p2)

        self.shape_matching_weight = 1000
        self.shape_recon_weight = 1000  
        self.latent_weight = 100
       # self.manifold_weight = 100
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * (2048-128))
        )

    def _group_points(self, nbrs, center, B, G):
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        # group points
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        B, G, _ = center.shape
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
      
        feat  = self.encoder_ueps(rebuild_points.view(B, -1, 3).permute(0,2,1).contiguous())

        pred = self.generator(feat).contiguous()

        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        #shape_recon_loss = self.shape_recon_weight * self.shape_criterion_info(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        
        ##! 
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()

        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        ##* 
        feat_recon = self.encoder_ueps(nbrs_pred.view(B, -1, 3).detach()) 

        
       latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        
       latent_recon_loss = self.latent_weight * self.latent_criterion(g1, g1_recon)
        normal consistency constraint
       manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss #+ manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss ,latent_recon_loss#, manifold_penalty

    def forward(self, partial, n_points=None, record=False):
        # group points
        B, _, _ = partial.shape

       #原始没有转置
        partial=partial.permute(0,2,1)


        feat= self.encoder_ueps(partial)
        pred = self.generator(feat).contiguous()


        return pred


    def forward(self, partial_cloud):
            """
            Args:
                partial_cloud: (B, N, 3)
            """
            # Encoder
            feat, patch_xyz, patch_feat = self.forward_encoder(partial_cloud)
            ##!嵌入层
            # Decoder
            pred_pcds = self.forward_decoder(feat, partial_cloud, patch_xyz, patch_feat)

            return pred_pcds
    
    

       

    
                                                                                       
if __name__ == '__main__':

    model = P2C()
    model = model.cuda()
    # print(model)

    x = torch.rand(2, 2048, 3)
    x = x.cuda()
    y = model(x)
    for i in y:
        print('y:',i.shape)
