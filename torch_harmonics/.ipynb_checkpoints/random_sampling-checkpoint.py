import numpy as np
import torch

class RandomSphericalSampling:
    r"""
    Defines a module for sampling a (uniformly) random set of measurement points from a grid. 

    [1] L. Lingsch, M. Michelis, E. de Bezenac, S. M. Perera, R. K. Katzschmann, S. Mishra; 
    Beyond Regular Grids: Fourier-Based Neural Operators on Arbitrary Domains; ICML 2024
    """
    def __init__(self, number_points_x, number_points_y):
        # the data must be equispaced
        self.number_points_x = number_points_x
        self.number_points_y = number_points_y
        np.random.seed(0)

    def random_points_on_sphere(self, n):
        r"""
        This function generates points within a 2x2x2 cube, centered at the origin. 
        Points with a radius<=1 are projected to a sphere with radius 1, centered at the origin.
        Points with a radius > 1 are excluded. The newly generated random points are used to select
        the closest points from the original grid, removing any duplicate points in this selection.

        Inputs:
        class variables
        n; approximate number of points to be selected (doubled, as about half the randomly generated points must be removed)
        Outputs:
        theta_index; vector indices of the original grid points to be selected along polar angle
        phi_index; vector indices of the original grid points to be selected along azimuthal angle
            >> used for selecting the points from the original data
        theta_angle; vector of polar angles for points, ranging from 0 to pi
        phi_angle; vector of azimuthal angles for points, ranging from 0 to 2*pi
        """
        # Double the number of points to be selected, as approximately half will not be valid
        n = n*2
        
        # Generate random points in 3D space
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.random.uniform(-1, 1, n)

        # remove all points with radius greater than 1 (slightly less than half of all points)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        mask = magnitude <= 1.0
        magnitude_filtered = magnitude[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]

        # Normalize the points to lie on the unit sphere
        x /= magnitude_filtered
        y /= magnitude_filtered
        z /= magnitude_filtered

        # Return the points on the sphere
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) + np.pi

        theta = np.floor(theta*self.number_points_y / np.pi)
        phi = np.floor(phi*self.number_points_x / (2*np.pi))

        # remove duplicate points (there are about 2% duplicates, generally)
        # Combine phi and theta into a 2D array
        positions = np.column_stack((phi, theta))
        # Remove duplicate positions
        unique_positions = np.unique(positions, axis=0)

        # Extract the cleaned phi and theta vectors
        phi_index = unique_positions[:, 0]
        theta_index = unique_positions[:, 1]

        phi_angle = torch.from_numpy(phi_index) / self.number_points_x * 2 * torch.pi
        theta_angle = torch.from_numpy(theta_index) / self.number_points_y * torch.pi

        self.theta_index = theta_index
        self.phi_index = phi_index

        return theta_index, phi_index, theta_angle.to(torch.float), phi_angle.to(torch.float)

    def random_sets_on_sphere(self, n, batch_size):
        theta_indices = []
        phi_indices = []
        thetas = []
        phis = []

        for _ in range(batch_size):
            theta_index, phi_index, theta, phi = self.random_points_on_sphere(n)
            theta_indices.append(theta_index)
            phi_indices.append(phi_index)
            thetas.append(theta)
            phis.append(phi)

        
        max_cols = min(matrix.size for matrix in theta_indices)
        
        padded_vectors = torch.zeros(len(theta_indices), max_cols, dtype=torch.float32, requires_grad=True)
        theta_index = np.zeros((len(theta_indices), max_cols))
        phi_index = np.zeros((len(theta_indices), max_cols))
        theta = torch.zeros_like(padded_vectors)
        phi = torch.zeros_like(padded_vectors)
        
        
        for i in range(batch_size):
            theta_index[i, :max_cols] = theta_indices[i][ :max_cols]
            phi_index[i, :max_cols] = phi_indices[i][ :max_cols]
            theta[i, :max_cols] = thetas[i][ :max_cols]
            phi[i, :max_cols] = phis[i][ :max_cols]
            
        # for i in range(batch_size):
        #     n_points = theta_indices[i].size
        #     theta_index[i, :n_points] = theta_indices[i]
        #     phi_index[i, :n_points] = phi_indices[i]
        #     theta[i, :n_points] = thetas[i]
        #     phi[i, :n_points] = phis[i]

        return theta_index, phi_index, theta, phi
        

    def get_random_sphere_data(self, data, thetas, phis):
        
        batch_size = thetas.shape[0]
        num_points = thetas.shape[1]
        
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_points)
        
        data_sparse = data[batch_indices,:,thetas,phis]
        
        return data_sparse.permute(0,2,1)
    