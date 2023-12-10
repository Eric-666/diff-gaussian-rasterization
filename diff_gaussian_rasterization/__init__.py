#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    backward_mask
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        backward_mask
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        backward_mask
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer,backward_mask)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer,backward_mask = ctx.saved_tensors

        #对所有的属性进行mask
        colors_precomp_remian = torch.empty(0)
        cov3Ds_precomp_remain = torch.empty(0)
        if colors_precomp.shape[0]!=0:
            colors_precomp_remian = colors_precomp[backward_mask]
        means3D_remian = means3D[backward_mask]
        scales_remain = scales[backward_mask]
        rotations_remain = rotations[backward_mask]
        if cov3Ds_precomp.shape[0]!=0:
            cov3Ds_precomp_remain = cov3Ds_precomp[backward_mask]
        radii_remain = radii[backward_mask]
        sh_remain = sh[backward_mask]
        num_rendered_remain = means3D.shape[0]

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D_remian, 
                radii_remain, 
                colors_precomp_remian, 
                scales_remain, 
                rotations_remain, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp_remain, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh_remain, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered_remain,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        #返回完整的梯度
        grad_means3D_full = torch.zeros_like(means3D)
        grad_means2D_full = torch.zeros_like(means3D)
        grad_sh_full = torch.zeros_like(sh)
        grad_colors_precomp_full = torch.zeros_like(colors_precomp)
        grad_opacities_full = torch.zeros((means3D.shape[0],0)).cuda()
        grad_scales_full = torch.zeros_like(scales)
        grad_rotations_full = torch.zeros_like(rotations)
        grad_cov3Ds_precomp_full = torch.zeros_like(cov3Ds_precomp)

        grad_means3D_full[backward_mask==1] = grad_means3D
        grad_means2D_full[backward_mask==1] = grad_means2D
        grad_sh_full[backward_mask==1] = grad_sh
        if colors_precomp.shape[0]!=0:
            grad_colors_precomp_full[backward_mask==1] = grad_colors_precomp 
        grad_opacities_full[backward_mask==1] = grad_opacities
        grad_scales_full[backward_mask==1] = grad_scales
        grad_rotations_full[backward_mask==1] = grad_rotations
        if cov3Ds_precomp.shape[0]!=0:
            grad_cov3Ds_precomp_full[backward_mask==1] = grad_cov3Ds_precomp

        grads = (
            grad_means3D_full,
            grad_means2D_full,
            grad_sh_full,
            grad_colors_precomp_full,
            grad_opacities_full,
            grad_scales_full,
            grad_rotations_full,
            grad_cov3Ds_precomp_full,
            None,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, backward_mask = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            backward_mask
        )

