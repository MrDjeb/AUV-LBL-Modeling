import numpy as np


class UnderwaterRobot:
    def __init__(self, x0=0, y0=0, z0=0, velocity=1.0, heading=0, heading_error=0, velocity_error=0):
        self.x = x0
        self.y = y0
        self.z = z0
        self.velocity = velocity
        self.heading = np.radians(heading)
        self.x_ssp = x0
        self.y_ssp = y0
        self.ssp_velocity = velocity
        self.ssp_total_error_heading = 0.0
        self.heading_error = np.radians(heading_error)
        self.velocity_error = velocity_error
        self.x_gans = x0
        self.y_gans = y0
        
    def update_position(self, dt, new_heading=None):
        if new_heading is not None:
            self.heading = np.radians(new_heading)
        
        # Обновление истинных координат
        vx = self.velocity * np.cos(self.heading)
        vy = self.velocity * np.sin(self.heading)
        self.x += vx * dt
        self.y += vy * dt
        
    def update_ssp(self, ssp_period):
        # Расчет параметров с ошибками для SSP
        #self.ssp_total_error_heading = self.ssp_total_error_heading + np.radians(self.heading_error) 
        self.ssp_total_error_heading =   np.radians(self.heading_error ) +np.radians(np.random.normal(0, 0.5) )
        ssp_heading = self.heading + self.ssp_total_error_heading

        self.ssp_velocity = self.velocity + self.velocity_error + np.random.normal(0, 0.05)     
        #print(self.x_ssp, self.y_ssp)

        # Обновление координат SSP
        vx_ssp = self.ssp_velocity * np.cos(ssp_heading)
        vy_ssp = self.ssp_velocity * np.sin(ssp_heading)
        self.x_ssp += vx_ssp * ssp_period
        self.y_ssp += vy_ssp * ssp_period

        #self.x_ssp = self.x
        #self.y_ssp = self.y

    def update_gans(self, x_gans, y_gans):
        self.x_gans = x_gans
        self.y_gans = y_gans

    
    def get_measurements(self, beacons, noise_std=0):

        exact_dists = np.linalg.norm(beacons - np.array([self.x, self.y, self.z]), axis=1)
        noise = np.random.normal(0, noise_std, size=exact_dists.shape)
        noisy_dists = exact_dists + noise
        projection = np.sqrt(noisy_dists**2 - (self.z - beacons[:, 2])**2)

        # Вычисляем веса как обратные величины дисперсии шума
        if noise_std == 0:
            weights = np.ones_like(noisy_dists)
        else:
            weights = 1 / (noise_std ** 2)

        return noisy_dists, projection, weights