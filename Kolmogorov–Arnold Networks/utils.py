import torch
from torch.utils.data import Dataset 
import numpy as np


def extend_grid(grid, k_extend=0):
    """
    Args:
        - grid (torch.tensor of shape (in_dim, grid_size+1)): the grid to be extended
        - k_extend (int): spline_order
    """
    
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    
    return grid


def b_splines(x, grid, spline_order):
        """
        Calculate the value of b-splines at x
        Args:
            x: torch.tensor of shape (batch_size, in_dim)
        Return:
            bases: torch.tensor of shape (batch_size, in_dim, grid_size+spline_order)
        """

        x = x.unsqueeze(-1) 
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype) 
        for k in range(1, spline_order+1):
            bases = (
                (x - grid[:, :-(k+1)])
                / (grid[:, k:-1] - grid[:, :-(k+1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, (k+1):] - x)
                / (grid[:, (k+1):] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
            
        bases = torch.nan_to_num(bases)
        
        return bases.contiguous()


def curve2coef(x, y, grid, spline_order):
        """
        Calculate c_i coefficients in eq (17), page 28 in the original paper given the values of x and spline(x) using least squares
        Args:
            - x: torch.tensor of shape (batch_size, in_dim)
            - y: torch.tensor of shape (batch_size, in_dim, out_dim)
        Return:
            result: torch.tensor of shape (in_dim, out_dim, grid_size+spline_order)
        """
        A = b_splines(x, grid, spline_order).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(0, 2, 1)
        
        return result.contiguous()


def coef2curve(x, grid, spline_order, coef):
    """
    Compute the values of activation functions at x
    Args:
        - x: torch.tensor of shape (batch_size, in_dim)
        - grid: torch.tensor of shape (in_dim, grid_size+1+2*spline_order) 
        - coef: torch.tensor of shape (in_dim, out_dim, grid_size+spline_order)
    Return:
        y: torch.tensor of shape (batch_size, in_dim, out_dim)
    """
    
    y = b_splines(x, grid, spline_order).unsqueeze(2) * coef.unsqueeze(0) # shape=(batch_size, in_dim, out_dim, grid_size+spline_order)
    y = torch.sum(y, dim=-1) # shape=(batch_size, in_dim, out_dim)

    return y


def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1., 1.],
                   train_num=100, 
                   test_num=100,
                   ):
    """
    Args:
        - f : the function used to create the dataset
        - ranges (list or np.array of shape (2,) or (n_var, 2)): the range of input variables
        - train_num (int): the number of training samples
        - test_num (int): the number of test samples
    Return:
        - train_dataset (torch.utils.data.Dataset): train dataset
        - test_dataset (torch.utils.data.Dataset): test dataset
    """
    
    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = torch.rand(train_num,)*(ranges[i, 1]-ranges[i, 0]) + ranges[i, 0]
        test_input[:, i] = torch.rand(test_num,)*(ranges[i, 1]-ranges[i, 0]) + ranges[i, 0]
                
    train_label = f(train_input)
    test_label = f(test_input)
    
    class MyDataset(Dataset):
        def __init__(self, input_data, labels):
            super().__init__()
            self.input_data = input_data
            self.labels = labels
            
        def __len__(self):
            return len(self.input_data)
            
        def __getitem__(self, idx):
            return self.input_data[idx], self.labels[idx]

    train_dataset = MyDataset(train_input, train_label)
    test_dataset = MyDataset(test_input, test_label)

    return train_dataset, test_dataset
    