import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Propagation:
    """ Base class for the mathematical models simulating short range signal 
    propagation.
    """
    def __init__(self):
        pass

    def __call__(self, rho, phi, theta=None):
        """ Return the received signal strength (normalized for the moment) 
        based on the reception radius and misalignment angle. 
        =======================================================================
        - Args:
            rho [float] : the distance to the target position in centimeters.
            phi [float] : the misalignment in radians of the receiver direction 
                and the target position.
        - Returns:
            The normalized signal strength of the reception [float].
        =======================================================================
        """
        raise NotImplementedError
    
    def plot(self, max_rad=5, sensor_name=None):
        """ Illustrates the polar plot of a sector coverage.
        =============================================================
        -Args:
            max_rad [float] : maximum coverage range (to be ploted) 
                It is expressed in centimeters.
            sensor_name [str] : Name of type of the sensor to include 
                it in the figure title.
        =============================================================
        """
        Rvals = np.linspace(0, max_rad, 200)
        theta_vals = np.radians(np.linspace(0, 360, 360))
        Rgrid, theta_grid = np.meshgrid(Rvals, theta_vals)
        direction = 0
        values = np.zeros([Rgrid.shape[0], Rgrid.shape[1]])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                theta = theta_grid[i,j]
                phi = np.abs(direction - theta)
                if phi > np.pi: 
                    phi = 2 * np.pi - phi
                values[i, j] = self(Rgrid[i,j], phi) 
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        line = ax.contourf(theta_grid, Rgrid, values, levels=50,\
                        cmap=plt.get_cmap('Reds'))   
        fig.colorbar(line, ax=ax)
        if sensor_name is not None:
            ax.set_title('Coverage of {} for a sector.'.format(sensor_name))
        else:
            ax.set_title('Sector Coverage.')
        plt.show()
        
    def plot_directivity(self, max_rad=5, sensor_name=None):
        """ Illustrates the polar plot of a sector coverage.
        =============================================================
        -Args:
            max_rad [float] : maximum coverage range (to be ploted) 
                It is expressed in centimeters.
            sensor_name [str] : Name of type of the sensor to include 
                it in the figure title.
        =============================================================
        """
        directions = [0, np.pi/2, np.pi, 1.5*np.pi]#[0]
        theta_vals = np.radians(np.linspace(0, 360, 360))
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        for direction in directions:
            phi_vals = np.abs(direction - theta_vals)
            phi_vals[phi_vals > np.pi] =  2 * np.pi - phi_vals[phi_vals > np.pi]
            
            # import pdb; pdb.set_trace()
            ax.plot(theta_vals, [self(0.01, th) for th in phi_vals])
            
            # if sensor_name is not None:
            #     ax.set_title('Coverage of {} for a sector.'.format(sensor_name))
            # else:
            #     ax.set_title('Sector Coverage.')
        plt.show()

class ExpDecayPropagation(Propagation):
    """ Simplified signal propagation using the exponential decaying 
    of the signal of both radius and phi. Both terms are combined as 
    a product.
    """
    def __init__(self, rho_att=0.3, phi_att=1):
        self.rho_att = rho_att
        self.phi_att = phi_att

    def __call__(self, rho, phi):
        return np.exp(-self.rho_att * rho)  * np.exp(-self.phi_att * phi)
        # return np.exp(-self.rho_att * rho) * np.exp(-.75* phi ** 2)# 1.5 DS, 0.75 LS
        # return signal

class RSSI_Propagation(Propagation):
    """ 
    """
    def __init__(self, noise_sigma=0.05):
        self.rssi_0 = -69
        self.n = 2.
        self.noise_sigma = noise_sigma
        self.buffer = [0.0] * 3

    def __call__(self, rho, phi):
        rho = max(rho, 1e-3)
        #* Simulate measured noisy RSSI
        rssi = self.rssi_0 - 10 * self.n * np.log10(rho) + np.random.randn() * self.noise_sigma 
        # Normalize (suppose max dist 10 meters)
        # rssi_max = self.rssi_0 - 10 * self.n
        # rssi_min = self.rssi_0 - 10 * self.n * np.log10(0.1)
        estim_dist = 10 ** ((-rssi + self.rssi_0) / (10*self.n))
        self.buffer.append(estim_dist)
        self.buffer.pop(0)
        estim_dist = np.array([0.1, 0.3, 0.6]).dot(np.array(self.buffer))
        return estim_dist / 10