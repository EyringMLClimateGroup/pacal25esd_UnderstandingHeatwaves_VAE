import torch
from torch import nn


class SpatialTemporalEncoder(nn.Module):
    def __init__(self, input_dim, input_shape, hidden_dim, kernel_size, stride, padding):
        super(SpatialTemporalEncoder, self).__init__()
        
        self.input_shape = input_shape        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding     
        
        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(self.input_dim, self.hidden_dim // (2*2*2), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(),            
            nn.BatchNorm3d(self.hidden_dim // (2*2*2))
        )
        self.max_pool1 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True) 
        
        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // (2*2*2), self.hidden_dim // (2*2), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.LeakyReLU(),            
            nn.BatchNorm3d(self.hidden_dim // (2*2))
        )
        self.max_pool2 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True) 
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // (2*2), self.hidden_dim // 2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // 2),
            nn.LeakyReLU()
        )
        self.max_pool3 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True) 
            
        self.enc_conv4 = nn.Sequential(
            nn.Conv3d(self.hidden_dim // 2, self.hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), 
            nn.BatchNorm3d(self.hidden_dim),
        )
        self.max_pool4 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), return_indices=True) 
        
        
    def forward(self, x):
        x = self.enc_conv1(x)
        x, idx1 = self.max_pool1(x) 
        x = self.enc_conv2(x)
        x, idx2 = self.max_pool2(x)
        x = self.enc_conv3(x)
        x, idx3 = self.max_pool3(x)
        x = self.enc_conv4(x)
        x, idx4 = self.max_pool4(x)
                
        return x, [idx1, idx2, idx3, idx4]


class SpatialTemporalDecoder(nn.Module):
    def __init__(self, hidden_dim, input_dim, input_shape, kernel_size, stride, padding):
        super(SpatialTemporalDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim        
        self.input_dim = input_dim       
        self.input_shape = input_shape        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding     
        
        # self.unflatten_decoder = nn.Unflatten(dim=1, unflattened_size=(self.hidden_dim, input_shape[0] // 16, input_shape[1] // 16, input_shape[2] // 16))
                                                
        self.max_unpool4 = nn.MaxUnpool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)) 
        self.dec_conv4 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim, self.hidden_dim // 2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // 2),
            nn.LeakyReLU() 
        )
        
        self.max_unpool3 = nn.MaxUnpool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)) 
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // 2, self.hidden_dim // (2*2), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // (2*2)),
            nn.LeakyReLU() 
        )
        
        self.max_unpool2 = nn.MaxUnpool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // (2*2), self.hidden_dim // (2*2*2), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.hidden_dim // (2*2*2)),
            nn.LeakyReLU() 
        )

        self.max_unpool1 = nn.MaxUnpool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)) 
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose3d(self.hidden_dim // (2*2*2), self.input_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm3d(self.input_dim)        
        )
        
    def forward(self, x, indices):
        x = self.max_unpool4(x, indices[3])
        x = self.dec_conv4(x)
        x = self.max_unpool3(x, indices[2])
        x = self.dec_conv3(x)
        x = self.max_unpool2(x, indices[1])
        x = self.dec_conv2(x)
        x = self.max_unpool1(x, indices[0])
        x = self.dec_conv1(x)
                
        return x
        
        
class VAEConv3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, input_shape, kernel_size, stride, padding, apply_sigmoid=False):
        super(VAEConv3D, self).__init__()
        
        # Initialize
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding        
        self.apply_sigmoid = apply_sigmoid
        
        # ENCODER
        self.spatial_temporal_encoder = SpatialTemporalEncoder(self.input_dim,self.input_shape,  self.hidden_dim, self.kernel_size, self.stride, self.padding)
        self.flatten_dim, self.tl, self.hl, self.wl = self._calculate_flatten_dim()
        self.flatten_encoder = nn.Flatten(start_dim=1)
                
        # LATENT SPACE
        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        
        # DECODER
        self.fc_decoder = nn.Linear(self.latent_dim, self.flatten_dim)
        self.unflatten_decoder = nn.Unflatten(dim=1, unflattened_size=(self.hidden_dim, self.tl, self.hl, self.wl))
        self.spatial_temporal_decoder = SpatialTemporalDecoder(self.hidden_dim, self.input_dim, self.input_shape, self.kernel_size, self.stride, self.padding)
        self.final_activation = nn.Sigmoid()
            
        
    def _calculate_flatten_dim(self):
        
        t, h, w = self.input_shape  #  Time, Height, Width
    
        # Calculate output size for each convolution layer in self.spatial_temporal_encoder
        for layer in self.spatial_temporal_encoder.children():
            if isinstance(layer, nn.Sequential):
                for l in layer:
                    if isinstance(l, nn.Conv3d):
                        t = (t + 2*l.padding[0] - l.dilation[0]*(l.kernel_size[0] - 1) - 1) // l.stride[0] + 1
                        h = (h + 2*l.padding[1] - l.dilation[1]*(l.kernel_size[1] - 1) - 1) // l.stride[1] + 1
                        w = (w + 2*l.padding[2] - l.dilation[2]*(l.kernel_size[2] - 1) - 1) // l.stride[2] + 1
                    elif isinstance(l, nn.MaxPool3d):
                        t = (t + 2*l.padding[0] - l.dilation*(l.kernel_size[0] - 1) - 1) // l.stride[0] + 1
                        h = (h + 2*l.padding[1] - l.dilation*(l.kernel_size[1] - 1) - 1) // l.stride[1] + 1
                        w = (w + 2*l.padding[2] - l.dilation*(l.kernel_size[2] - 1) - 1) // l.stride[2] + 1
                    else:
                        pass
            elif isinstance(layer, nn.MaxPool3d):
                
                t = (t + 2*layer.padding[0] - layer.dilation*(layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
                h = (h + 2*layer.padding[1] - layer.dilation*(layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
                w = (w + 2*layer.padding[2] - layer.dilation*(layer.kernel_size[2] - 1) - 1) // layer.stride[2] + 1

        return self.hidden_dim*t*h*w, t, h, w
    
    
    def encode(self, x):
        h, indices = self.spatial_temporal_encoder(x)
        h = self.flatten_encoder(h) 
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar, indices
    
    
    def reparameterize(self, mu, logvar):
        epsilon = 1e-6  # Variance floor
        std = torch.sqrt(torch.exp(logvar).clamp(min=epsilon))  # Apply variance floor
        eps = torch.randn_like(std)
        return eps*std + mu
    
    
    def decode(self, z, indices):
        z = self.fc_decoder(z)
        z = self.unflatten_decoder(z)
        z = self.spatial_temporal_decoder(z, indices)
        if self.apply_sigmoid:
            z = self.final_activation(z)
        return z    
    
    
    def forward(self, x):
        mu, logvar, indices = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, indices)
        return decoded, mu, logvar, z
    