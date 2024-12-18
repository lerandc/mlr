import torch

from torch import nn
from torch.special import erf
from functools import reduce

def bandlimit(x, offset, width):
    return (0.5+0.5*erf(width*(x + offset)))*(0.5-0.5*erf(width*(x - offset)))

class CorrelationKernelMLP(nn.Module):
    """
    Computes the kernel function from R^D -> R
    """
    def __init__(self, device, d=2, N_hidden_layers=2, hidden_dim=512,):
        super().__init__()

        self.device = device
        self.N_hidden_layers = N_hidden_layers
        self.d = d
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(d, hidden_dim, device=device), nn.ReLU()])

        self.layers.extend(
            reduce(
                lambda x, y: x + y,
                [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for i in range(N_hidden_layers)]
            )
        )

        self.layers.append(
            nn.Linear(hidden_dim, 1, device=device)
        )
        self.to(device)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
    
    def g2(self, R, K):
        # g2  - 1 = -f(x)*f(-x) /(f(0) * f(0))
        norm = self.forward(torch.zeros((1,self.d), device=self.device))
        C_r = self.forward(R)*self.forward(-R)
        C_r = C_r.reshape(self.d*(K,))

        return torch.ones_like(C_r) - C_r / (norm * norm)
    
    def s2(self, x, rho, K):
        # Sk = 1 + (rho) * FT[g2 - 1] | final subtraction carries sign from g2
        norm = self.forward(torch.zeros((1,self.d), device=self.device))
        fx = self.forward(x)
        rx = self.forward(-x)
        C_r = fx * rx #self.forward(x)*self.forward(-x)
        C_r = C_r.reshape(self.d*(K,))

        # C_r = torch.fft.ifftshift(C_r)
        res = (torch.fft.rfftn(C_r, norm='forward').real)

        res = (rho / (norm * norm)) * res

        return torch.ones_like(res) - res
    
    def s3(self, r_xy, r_yz, rho, K):

        """
        R_xy is N x d
        R_yz is N x d

        C_p/n.. will be length N arrays where N = K ** d
        
        """
        norm = self.forward(torch.zeros((1,self.d), device=self.device))

        r_zx = (r_yz[:, None, :] - r_xy[None, :, :]).reshape((-1, self.d))

        C_pxy = self.forward(r_xy)
        C_pyz = self.forward(r_yz).T
        C_pzx = self.forward(r_zx).reshape(((C_pxy.shape[0], C_pyz.shape[1])))

        C_nxy = self.forward(-1.0*r_xy)
        C_nyz = self.forward(-1.0*r_yz).T
        C_nzx = self.forward(-1.0*r_zx).reshape((C_nxy.shape[0], C_nyz.shape[1]))

        ## 2 body components
        C_r2 = (-1.0/(norm*norm))*(C_pxy*C_nxy + C_pyz*C_nyz + C_pzx*C_nzx)
        C_r2 = C_r2.reshape(2*self.d*(K,))

        ## 3 body components
        C_r3 = (1.0/(norm*norm*norm))*(C_pxy*C_pyz*C_pzx + C_nxy*C_nyz*C_nzx)
        C_r3 = C_r3.reshape(2*self.d*(K,))

        res = torch.fft.rfftn(C_r2 + C_r3).real

        return torch.ones_like(res) + rho * res 
    

class BoundedCorrelationKernelMLP(CorrelationKernelMLP):

    def __init__(self, *args, init_mode=None, bandlimiter=None, **kwargs):
        super().__init__(*args, **kwargs)

        for m in self.modules():
            match m:
                case torch.nn.Linear():
                    match init_mode:
                        case 'normal':
                            torch.nn.init.normal_(m.weight)
                        case 'xuniform':
                            torch.nn.init.xavier_uniform_(m.weight)
                        case 'xnormal':
                            torch.nn.init.xavier_normal_(m.weight)
                        case 'kuniform':
                            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                        case 'knormal':
                            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        case 'orthogonal':
                            torch.nn.init.orthogonal_(m.weight)
                        case _:
                            pass
                case _:
                    pass

        self.bandlimiter = bandlimiter
        


    def forward(self, x):
        x = super().forward(x)
        x = 2*torch.nn.functional.sigmoid(x) - 1.0 # bound to [-1,1]

        if self.bandlimiter is None or (len(x) != len(self.bandlimiter)):
            return x
        else:
            res = x * self.bandlimiter[:, None]
            return res
