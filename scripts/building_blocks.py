import numpy as np
import torch
import torch.nn as nn

class LinearProjector(nn.Module):
    """
    This layer aggregates the pixels along one dimension (3D representation -> 2D representation).
    Note that this projection alone leads to a huge loss of information.
    
    The linear projector is supposed to be isotrope (it is applied identically along the 3 dimensions)
    and is forced to be an even function (to bring invariance to reflections and 90° rotations)
    """
    
    def __init__(self, width, C_out, isotrope=False):
        """
        * width: number of pixels along the 3 dimensions (should be the same)
        * C_out: number of output channels
        """
        super().__init__()
        
        self.width = width
        self.C_out = C_out
        self.isotrope = isotrope
        
        if isotrope:
            l = width // 2

            if width % 2 == 0:
                w_ini = np.zeros(shape=(l, C_out))
            else:
                w_ini = np.zeros(shape=(l+1, C_out))
        else:
            w_ini = np.zeros(shape=(width, C_out))

        self.w = nn.Parameter(
            torch.from_numpy(w_ini).type(torch.float32),
            requires_grad=True
        )
        # Test to check if the projection is good or not
        self.w = nn.init.xavier_uniform_(self.w)
        
        self.bias = nn.Parameter(
            torch.from_numpy(np.zeros(shape=(1, C_out, 1, 1))).type(torch.float32),
            requires_grad=True
        )
        self.bias = nn.init.zeros_(self.bias)
        
    def forward(self, x, axis='x'):
        
        if self.isotrope:
            if self.width % 2 == 0:
                w = torch.cat((self.w, torch.flip(self.w, dims=(0,))), dim=0)
            else:
                w = torch.cat((self.w[:-1,:], torch.flip(self.w, dims=(0,))), dim=0)
        else:
            w = self.w
            
        if axis == 'x':
            out = torch.einsum('bxyz,xc -> bcyz', x, w) + self.bias[:,:,:,:]
        elif axis == 'y':
            out = torch.einsum('bxyz,yc -> bcxz', x, w) + self.bias[:,:,:,:]
        elif axis == 'z':
            out = torch.einsum('bxyz,zc -> bcxy', x, w) + self.bias[:,:,:,:]
            
        return out
    
class ConvertToQuaternions(nn.Module):
    """
    This layer converts 3D images to a 2D-quaternionic images
    based on the LinearProjector
    """
    
    def __init__(self, width, upsample_factor=1., isotrope=False, device='cuda'):
        """
        Currently, C_out is fixed to the value of width in order to add local dependencies
        in the real part of the quaternions.
        """
        super().__init__()
        
        self.device = device
        self.isotrope = isotrope
        
        if upsample_factor == 1.:
            self.upsample = None
        else:
            width = int(width*upsample_factor)
            self.upsample = nn.Upsample(size=width, mode='trilinear').to(device)
            
        self.width = width
        
        # Will learn non-local features
        if isotrope:
            self.non_local_projector = LinearProjector(width, width).to(device)
        else:
            self.non_local_projector = {'x': LinearProjector(width, width).to(device),
                                        'y': LinearProjector(width, width).to(device),
                                        'z': LinearProjector(width, width).to(device)}
        
        # Will learn local features
        self.local_projector = nn.Conv3d(in_channels=1, out_channels=1, 
                                         kernel_size=int(2.*upsample_factor+1), padding='same')
        
    def forward(self, x):
        """
        x: (b, W, W, W)
        out: (b, C_out, W, W, 4) with C_out == W
        """
        
        assert x.shape[1] == x.shape[2]
        assert x.shape[2] == x.shape[3]
        
        if self.upsample:
            x = self.upsample(x[:,None,:,:,:])[:,0,:,:,:]
        
        out = torch.zeros((x.shape[0], self.width, self.width, self.width, 4), device=self.device)
        #out[:,:,:,:,0] = self.local_projector(x[:,None,:,:,:])[:,0,:,:,:]
        if self.isotrope:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:,axis+1] = self.non_local_projector(x, axis=proj)
        else:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:,axis+1] = self.non_local_projector[proj](x, axis=proj)
            
        return out
    
class QConv(nn.Module):
    """
    This layer implements a "Separable Quaternion Convolution"
    """
    
    def __init__(self, k, C_in, C_out, padding, device='cuda'):
        super().__init__()
        
        self.device = device
        
        self.conv_r = nn.Conv2d(in_channels=C_in, out_channels=C_out,
                                kernel_size=k, padding=padding).to(device)
        self.conv_x = nn.Conv2d(in_channels=C_in, out_channels=C_out,
                                kernel_size=k, padding=padding).to(device)
        self.conv_y = nn.Conv2d(in_channels=C_in, out_channels=C_out,
                                kernel_size=k, padding=padding).to(device)
        self.conv_z = nn.Conv2d(in_channels=C_in, out_channels=C_out,
                                kernel_size=k, padding=padding).to(device)
        
    def forward(self, x):
        """
        x: (b, C_in, W, W, 4)
        out: (b, C_out, W, W, 4)
        """
        
        # Compute the 4 convolutions
        r_out = self.conv_r(x[:,:,:,:,0]) # V_r * S_r
        x_out = self.conv_x(x[:,:,:,:,1]) # V_x * S_x
        y_out = self.conv_y(x[:,:,:,:,2]) # V_y * S_y
        z_out = self.conv_z(x[:,:,:,:,3]) # V_z * S_z
        
        # Compute the crossed terms
        out = torch.empty((r_out.shape[0], r_out.shape[1], r_out.shape[2], r_out.shape[3], 4), device=self.device)
        out[:,:,:,:,0] = r_out + x_out + y_out + z_out
        out[:,:,:,:,1] = x_out - r_out - z_out + y_out
        out[:,:,:,:,2] = y_out + z_out - r_out - x_out
        out[:,:,:,:,3] = z_out - y_out + x_out - r_out
        
        return out
    
class QLinear(nn.Module):
    """
    This layer implements a "Separable Quaternion" dense layer
    """
    
    def __init__(self, C_in, C_out, device='cuda'):
        super().__init__()
        
        self.device = device
        
        self.lin_r = nn.Linear(in_features=C_in, out_features=C_out).to(device)
        self.lin_x = nn.Linear(in_features=C_in, out_features=C_out).to(device)
        self.lin_y = nn.Linear(in_features=C_in, out_features=C_out).to(device)
        self.lin_z = nn.Linear(in_features=C_in, out_features=C_out).to(device)
        
    def forward(self, x):
        """
        x: (b, features_in, 4)
        out: (b, features_out, 4)
        """
        
        # Compute the 4 convolutions
        r_out = self.lin_r(x[:,:,0]) # V_r * S_r
        x_out = self.lin_x(x[:,:,1]) # V_x * S_x
        y_out = self.lin_y(x[:,:,2]) # V_y * S_y
        z_out = self.lin_z(x[:,:,3]) # V_z * S_z
        
        # Compute the crossed terms
        out = torch.empty((r_out.shape[0], r_out.shape[1], 4), device=self.device)
        out[:,:,0] = r_out + x_out + y_out + z_out
        out[:,:,1] = x_out - r_out - z_out + y_out
        out[:,:,2] = y_out + z_out - r_out - x_out
        out[:,:,3] = z_out - y_out + x_out - r_out
        
        return out

class QMaxPool2D(nn.Module):
    """
    This layer implements a 2D max pooling for quaternion feature maps
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, device='cuda'):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, 
                                 padding=padding, dilation=dilation).to(device)
        
    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, W_out, W_out, 4)
        """
        
        # Compute the 4 convolutions
        r_out = self.pool(x[:,:,:,:,0])
        x_out = self.pool(x[:,:,:,:,1])
        y_out = self.pool(x[:,:,:,:,2])
        z_out = self.pool(x[:,:,:,:,3])

        return torch.cat((r_out[:,:,:,:,None], x_out[:,:,:,:,None], 
                          y_out[:,:,:,:,None], z_out[:,:,:,:,None]), dim=4)
    
class QGlobalMaxPool2D(nn.Module):
    """
    This layer implements a global max pooling for quaternion feature maps
    """

    def __init__(self, device='cuda'):
        super().__init__()
        
        self.pool = nn.AdaptiveMaxPool2d((1,1)).to(device)

    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, 1, 1, 4)
        """
        
        # Compute the 4 convolutions
        r_out = self.pool(x[:,:,:,:,0])
        x_out = self.pool(x[:,:,:,:,1])
        y_out = self.pool(x[:,:,:,:,2])
        z_out = self.pool(x[:,:,:,:,3])

        return torch.cat((r_out[:,:,:,:,None], x_out[:,:,:,:,None], 
                          y_out[:,:,:,:,None], z_out[:,:,:,:,None]), dim=4)
    
class QUpsample(nn.Module):
    """
    This layer implements an upsampling for quaternion feature maps
    """

    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False, device='cuda'):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners).to(device)

    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, 2*W, 2*W, 4)
        """
        
        # Compute the 4 convolutions
        r_out = self.up(x[:,:,:,:,0])
        x_out = self.up(x[:,:,:,:,1])
        y_out = self.up(x[:,:,:,:,2])
        z_out = self.up(x[:,:,:,:,3])

        return torch.cat((r_out[:,:,:,:,None], x_out[:,:,:,:,None], 
                          y_out[:,:,:,:,None], z_out[:,:,:,:,None]), dim=4)    

class QChannelAttention(nn.Module):
    """
    This layer implements a "Quaternion Channel Attention"
    """
    
    def __init__(self, C_in, device='cuda'):
        super().__init__()
        
        self.device = device

        self.pool = QGlobalMaxPool2D()
        self.QLinear_comp = QLinear(C_in, 1)
        self.QLinear_exc = QLinear(1, C_in)

    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, W, W, 4)
        """
        
        # Compute the quaternionic global max pooling
        out = self.pool(x)
        
        # Compute the quaternionic linear layers
        u = self.QLinear_comp(out.view(out.shape[0], out.shape[1], 4))
        u = self.QLinear_exc(u.view(u.shape[0], 1, 4))

        # Compute the separable product
        r_out = x[:,:,:,:,0] * u[:,:,None,None,0]
        x_out = x[:,:,:,:,1] * u[:,:,None,None,1]
        y_out = x[:,:,:,:,2] * u[:,:,None,None,2]
        z_out = x[:,:,:,:,3] * u[:,:,None,None,3]

        out = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3], 4), device=self.device)
        out[:,:,:,:,0] = r_out# + x_out + y_out + z_out
        out[:,:,:,:,1] = x_out# - r_out - z_out + y_out
        out[:,:,:,:,2] = y_out# + z_out - r_out - x_out
        out[:,:,:,:,3] = z_out# - y_out + x_out - r_out

        return out
    
class QSpatialAttention(nn.Module):
    """
    This layer implements a "Quaternion Spatial Attention"
    """
    
    def __init__(self, k, C_in, n_groups, device='cuda'):
        super().__init__()
        
        self.C_in = C_in
        self.n_groups = n_groups
        self.device = device
        
        assert C_in % n_groups == 0

        # To compute U
        self.QConv_comp = QConv(k, C_in, 1, k//2, device)
        
        # To expand U for the different groups
        self.QConvs_exp = []
        for i in range(n_groups):
            self.QConvs_exp.append(QConv(k, 1, C_in//n_groups, k//2, device))
        

    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, W, W, 4)
        """
        
        # Compute U
        U = self.QConv_comp(x) # (b, 1, W, W, 4)
        
        parts = []
        for i in range(self.n_groups):
            U_exp = self.QConvs_exp[i](U) # (b, C_in//n_groups, W, W, 4)
            idx = i*(self.C_in//self.n_groups)
            parts.append(x[:,idx:idx+self.C_in//self.n_groups,:,:,:] * U_exp) # (b, C_in//n_groups, W, W, 4)

        return torch.concat(parts, dim=1)

class QBatchNorm(nn.Module):
    """
    This layer implements a "Quaternion Batch Normalization"
    """
    
    def __init__(self, C_in):
        super().__init__()
        
        self.eps = 1e-9
        
        # Mean scaler
        self.beta = nn.Parameter(
            torch.from_numpy(np.zeros(shape=(C_in, 4))).type(torch.float32),
            requires_grad=True
        )
        
        # Std scaler
        self.gamma = nn.Parameter(
            torch.from_numpy(np.ones(shape=(C_in,))).type(torch.float32),
            requires_grad=True
        )
        
    def forward(self, x):
        """
        x: (b, C, W, W, 4)
        out: (b, C, W, W, 4)
        """
        
        qbar = torch.mean(x, dim=(0,2,3), keepdim=False) # output is (C, 4)
        QV = torch.mean(torch.square(x - qbar[None, :, None, None, :]), dim=(0, 2, 3, 4), keepdim=False) # output is (C,)
        
        outputs = (1. / torch.sqrt(QV + self.eps))[None, :, None, None, None] * self.gamma[None, :, None, None, None]
        outputs = outputs * (x - qbar[None, :, None, None, :]) + self.beta[None, :, None, None, :]
        
        return outputs
    
class ReversedLinearProjector(nn.Module):
    
    def __init__(self, projector):
        super().__init__()
        
        self.projector = projector
        self.width = projector.width
        self.C_out = projector.C_out
        
    def forward(self, x, axis='x'):
        
        if self.width % 2 == 0:
            w = torch.cat((self.projector.w, torch.flip(self.projector.w, dims=(0,))), dim=0)
        else:
            w = torch.cat((self.projector.w[:-1,:], torch.flip(self.projector.w, dims=(0,))), dim=0)
        
        w = torch.linalg.pinv(w)
        bias = self.projector.bias
            
        if axis == 'x':
            out = torch.einsum('bxyz,xc -> bcyz', x - bias[:,:,:,:], w) 
        elif axis == 'y':
            out = torch.einsum('bxyz,yc -> bcxz', x - bias[:,:,:,:], w)
        elif axis == 'z':
            out = torch.einsum('bxyz,zc -> bcxy', x - bias[:,:,:,:], w)
            
        return out
    
class ConvertToVolume(nn.Module):
    """
    This layer converts 2D-quaternionic images to 3D-images
    """
    
    def __init__(self, converter, device='cuda'):
        super().__init__()
        
        self.converter = converter
        self.device = converter.device
        self.isotrope = converter.isotrope
        self.width = converter.width
        
        # Will learn non-local features
        if self.isotrope:
            self.non_local_projector = ReversedLinearProjector(projector = converter.non_local_projector).to(device)
        else:
            self.non_local_projector = {'x': ReversedLinearProjector(projector = converter.non_local_projector['x']).to(device),
                                        'y': ReversedLinearProjector(projector = converter.non_local_projector['y']).to(device),
                                        'z': ReversedLinearProjector(projector = converter.non_local_projector['z']).to(device)}
            
    def forward(self, x):
        """
        x: (b, C_out, W, W, 4) with C_out == W
        out: (b, W, W, W)
        """
        
        assert x.shape[1] == x.shape[2]
        assert x.shape[2] == x.shape[3]
        
        out = torch.empty((x.shape[0], self.width, self.width, self.width), device=self.device)
        if self.isotrope:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:] += self.non_local_projector(x[:,:,:,:,axis], axis=proj) / 3.
        else:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:] += self.non_local_projector[proj](x[:,:,:,:,axis], axis=proj) / 3.
            
        return out
    
class ConvertToVolumeLearnable(nn.Module):
    """
    This layer converts 2D-quaternionic images to 3D-images
    """
    
    def __init__(self, C_in, C_out, isotrope=False, device='cuda'):
        super().__init__()
        
        self.C_in = C_in
        self.C_out = C_out
        self.isotrope = isotrope
        self.device = device
        
        # Will learn non-local features
        if self.isotrope:
            self.non_local_projector = LinearProjector(C_in, C_out).to(device)
        else:
            self.non_local_projector = {'x': LinearProjector(C_in, C_out).to(device).to(device),
                                        'y': LinearProjector(C_in, C_out).to(device).to(device),
                                        'z': LinearProjector(C_in, C_out).to(device).to(device)}
            
    def forward(self, x):
        """
        x: (b, C_out, W, W, 4) with C_out == W
        out: (b, W, W, W)
        """
        
        assert x.shape[1] == x.shape[2]
        assert x.shape[2] == x.shape[3]
        
        out = torch.empty((x.shape[0], self.C_out, self.C_out, self.C_out), device=self.device)
        if self.isotrope:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:] += self.non_local_projector(x[:,:,:,:,axis], axis=proj) / 3.
        else:
            for axis, proj in enumerate(['x', 'y', 'z']):
                out[:,:,:,:] += self.non_local_projector[proj](x[:,:,:,:,axis], axis=proj) / 3.
            
        return out