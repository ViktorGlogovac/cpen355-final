import numpy as np

class KalmanFilter:
    def __init__(self):
        # Define the initial state (position and velocity)
        self.dt = 1.0  # time step
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])  # State transition matrix

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # Observation matrix: measuring (x, y) only

        self.P = np.eye(4) * 1000  # Initial covariance matrix (high uncertainty)
        self.R = np.eye(2) * 10  # Measurement noise covariance matrix for position (x, y)
        self.Q = np.eye(4)  # Process noise covariance matrix

        self.x = np.zeros((4, 1))  # Initial state: (center_x, center_y, velocity_x, velocity_y)

    def predict(self):
        """Predict the next state using the Kalman filter."""
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """Update the state based on new measurement (z)."""
        # Measurement residual: difference between actual measurement and predicted state
        y = z - np.dot(self.H, self.x)  # z is (center_x, center_y)

        # Compute the residual covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Compute the Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state estimate
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])  # Identity matrix
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x
