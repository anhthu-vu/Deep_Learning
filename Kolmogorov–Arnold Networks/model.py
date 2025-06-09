import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KANLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 grid_size=5,
                 spline_order=3,
                 base_func=nn.SiLU(),
                 grid_eps=0.02,
                 grid_range=[-1., 1.],
                 sp_trainable=True,
                 sb_trainable=True,
                ):

        """
        Args:
            - in_dim (int): input dimension
            - out_dim (int): output dimension
            - grid_size (int): the number of grid intervals
            - spline_order (int): the order of piecewise polynomial
            - base_func: residual function b(x)
            - grid_eps (float): used for update_grid
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of 
                samples. 0 < grid_eps < 1 interpolates between the two extremes.
            - grid_range (list/np.array of shape (2,)): the range of grids 
            - sp_trainable (bool): if True, scale_sp is trainable
            - sb_trainable (bool): if True, scale_base is trainable
        """
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size 
        self.spline_order = spline_order 
        self.grid_eps = grid_eps
        self.base_func = base_func
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=grid_size+1).unsqueeze(0).expand(self.in_dim, grid_size+1)
        grid = extend_grid(grid, k_extend=spline_order) # shape=(in_dim, grid_size+1+2*spline_order)        
        self.register_buffer("grid", grid)

        # Initialize c_i in eq (17) in page 18 in the original paper
        noises = (torch.rand(grid_size+1, in_dim, out_dim) - 0.5) * 0.3/grid_size # shape = (grid_size+1, in_dim, out_dim)
        self.coef = nn.Parameter(curve2coef(self.grid[:, spline_order:-spline_order].permute(1, 0), noises, self.grid, self.spline_order)) # shape=(in_dim, out_dim, grid_size+spline_order)

        self.scale_base = nn.Parameter((torch.rand(in_dim, out_dim)*2.-1.) * 1./np.sqrt(in_dim)).requires_grad_(sb_trainable) # w_b, shape=(in_dim, out_dim)
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * 1./np.sqrt(in_dim)).requires_grad_(sp_trainable) # w_s, shape=(in_dim, out_dim)

    
    def forward(self, x):
        """
        Args:
            x: torch.tensor of shape (batch_size, in_dim)
        Return:
            y: torch.tensor of shape (batch_size, out_dim)
        """
        
        base = self.base_func(x) 
        y = coef2curve(x, self.grid, self.spline_order, self.coef) # shape=(batch_size, in_dim, out_dim)
        y = self.scale_base.unsqueeze(0) * base.unsqueeze(-1) + self.scale_sp.unsqueeze(0) * y
        postacts = y.clone() # shape=(batch_size, in_dim, out_dim)
        y = torch.sum(y, dim=1)
        
        return y, postacts

    def update_grid(self, x):
        """
        Args:
            x: torch.tensor of shape (batch_size, in_dim)
        """
        
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0] # shape=(batch_size, in_dim)
        y_eval = coef2curve(x_pos, self.grid, self.spline_order, self.coef) # shape=(batch_size, in_dim, out_dim)
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0) # shape=(in_dim, num_interval+1)
            
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1).unsqueeze(0).to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(self.grid_size)
        self.grid.data = extend_grid(grid, k_extend=self.spline_order)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.spline_order)

    
class MultiKAN(nn.Module):
    def __init__(self,
                 width,
                 grid_size=3,
                 spline_order=3,
                 base_func=nn.SiLU(),
                 grid_eps=0.02,
                 grid_range=[-1., 1.],
                 sp_trainable=True,
                 sb_trainable=True,
                ):
        
        super().__init__()
        
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(width[:-1], width[1:]):
            self.layers.append(
                KANLayer(
                    in_dim,
                    out_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_func=base_func,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    sp_trainable=sp_trainable,
                    sb_trainable=sb_trainable,
                )
            )
        self.depth = len(self.layers)
        

    def forward(self, x):
        """
        Args:
            x: torch.tensor of shape (batch_size, in_dim)
        Return:
            x: torch.tensor of shape (batch_size, out_dim)
        """
        
        self.acts = [x]
        self.postacts = []
        
        for layer in self.layers:
            x, p_act = layer(x)
            self.acts.append(x)
            self.postacts.append(p_act)
        return x

    def update_grid(self, x):
        """
        Args:
            x: torch.tensor of shape (batch_size, in_dim)
        """
        
        for l in range(self.depth):
            _ = self.forward(x)
            self.layers[l].update_grid(self.acts[l])

    def reg(self, lamb_l1, lamb_entropy):
        """
        Args: 
            - lamb_l1 (float): L1 regularization coefficient
            - lamb_entropy (float): entropy regularization coefficient
        """
        
        postacts = self.postacts
        reg_ = 0.
        for l in range(self.depth):
            p_act_avg = torch.abs(postacts[l]).mean(dim=0)
            reg_l1 = torch.sum(p_act_avg)
            reg_entropy = -torch.sum(p_act_avg/(reg_l1+1e-8)*torch.log(p_act_avg/(reg_l1+1e-8)))
            reg_ += (lamb_l1*reg_l1 + lamb_entropy*reg_entropy)
        
        return reg_
        
    def fit(self, 
            writer,
            train_loader, test_loader,
            lamb=0., lamb_l1=0., lamb_entropy=0.,
            optimizer=None, steps=100, loss_fn=nn.MSELoss(),
            update_grid=True, grid_update_num=10, start_grid_update_step=-1, stop_grid_update_step=50,
            metric='accuracy',
           ):
        """
        Args:
            - writer (torch.utils.tensorboard.SummaryWriter): writer for logging metrics
            - train_loader (torch.utils.data.DataLoader): train data loader
            - test_loader (torch.utils.data.DataLoader): test data loader
            - lamb (float): overall regularization coefficient
            - lamb_l1 (float): L1 regularization coefficient
            - lamb_entropy (float): entropy regularization coefficient
            - optimizer (torch.optim): optimizer
            - steps (int): the number of training epochs
            - loss_fn: loss function
            - update_grid (bool): if True, update grids before stop_grid_update_step
            - grid_update_num (int): the number of grid updates before stop_grid_update_step
            - start_grid_update_step (int): no grid updates before this training step
            - stop_grid_update_step (int): no grid updates after this training step
            - metric (None or 'accuracy'): additional metric to compute 
        
        """
        
        pbar = tqdm(range(steps), ncols=100)
        if update_grid:
            grid_update_freq = int(stop_grid_update_step / grid_update_num)

        train_loss, test_loss = [], []
        train_acc, test_acc = [], []

        for epoch in pbar:
            loss_train, loss_test = 0., 0.
            if metric is not None:
                acc_train, acc_test = 0., 0.
            regular = 0.
            
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if update_grid and epoch < stop_grid_update_step and epoch >= start_grid_update_step:
                    if epoch % grid_update_freq == 0:
                        self.update_grid(x)

                pred = self.forward(x)
                pred = pred.squeeze()
                reg_ = self.reg(lamb_l1, lamb_entropy)
                regular += reg_.item()
                loss = loss_fn(pred, y) 
                loss_reg = loss + lamb*reg_
                optimizer.zero_grad()
                loss_reg.backward()
                optimizer.step()
                loss_train += loss.item()
                if metric is not None:
                    acc_train += (torch.argmax(pred, dim=-1) == y).float().mean().item()

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = self.forward(x)
                    pred = pred.squeeze()
                    loss_test += loss_fn(pred, y).item()
                    if metric is not None:
                        acc_test += (torch.argmax(pred, dim=-1) == y).float().mean().item()
                        
            if metric is not None:
                pbar.set_description(f"Epoch {epoch+1}: train_loss: %.2e | test_loss: %.2e | train_acc: %.2e | test_acc: %.2e | reg: %.2e | " % (loss_train/len(train_loader), loss_test/len(test_loader), acc_train/len(train_loader), acc_test/len(test_loader), regular/len(train_loader)))
                writer.add_scalar('train_loss', loss_train/len(train_loader), epoch)
                writer.add_scalar('test_loss', loss_test/len(test_loader), epoch)
                writer.add_scalar('train_acc', acc_train/len(train_loader), epoch)
                writer.add_scalar('test_acc', acc_test/len(test_loader), epoch)
                writer.add_scalar('regularization', regular/len(train_loader), epoch)
                
                train_loss.append(loss_train/len(train_loader))
                test_loss.append(loss_test/len(test_loader))
                train_acc.append(acc_train/len(train_loader))
                test_acc.append(acc_test/len(test_loader))
                
            else:
                pbar.set_description(f"Epoch {epoch+1}: train_loss: %.2e | test_loss: %.2e | reg: %.2e |" % (loss_train/len(train_loader), loss_test/len(test_loader), regular/len(train_loader)))
                writer.add_scalar('train_loss', loss_train/len(train_loader), epoch)
                writer.add_scalar('test_loss', loss_test/len(test_loader), epoch)
                writer.add_scalar('regularization', regular/len(train_loader), epoch)

                train_loss.append(loss_train/len(train_loader))
                test_loss.append(loss_test/len(test_loader))
                
        return train_loss, test_loss, train_acc, test_acc
                          
                