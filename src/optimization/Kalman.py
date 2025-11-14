import numpy as np


class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise):
        """
        dt: time step between frames.
        process_noise: scalar to scale process noise covariance.
        measurement_noise: scalar to scale measurement noise covariance.
        """
        self.dt = dt
        
        # State vector: [t_x, t_y, t_z, r_x, r_y, r_z, vt_x, vt_y, vt_z, vr_x, vr_y, vr_z]
        self.x = np.zeros((12, 1))  # You might initialize this with your first measurement.
        
        # Initial state covariance
        self.P = np.eye(12) * 1.0

        # State transition matrix F (constant velocity model)
        self.F = np.eye(12)
        # Translation update: t_new = t_old + dt * v_t
        self.F[0:3, 6:9] = dt * np.eye(3)
        # Rotation update: r_new = r_old + dt * v_r
        self.F[3:6, 9:12] = dt * np.eye(3)

        # Process noise covariance Q (tune this based on your system)
        self.Q = process_noise * np.eye(12)

        # Measurement matrix H
        # We directly measure tvec and rvec, so H picks the first 6 states.
        self.H = np.zeros((6, 12))
        self.H[0:3, 0:3] = np.eye(3)  # for tvec
        self.H[3:6, 3:6] = np.eye(3)  # for rvec

        # Measurement noise covariance R (tune this based on your sensor noise)
        self.R = measurement_noise * np.eye(6)

    def predict(self):
        # Predict the state ahead
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # z should be a 6x1 vector: [tvec; rvec]
        y = z - (self.H @ self.x)  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(12) - K @ self.H) @ self.P
        return self.x