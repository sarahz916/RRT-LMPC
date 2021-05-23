import numpy as np
import pdb

class system(object):
	"""docstring for system"""
	def __init__(self, x0, dt):
		self.x 	   = [x0]
		self.u 	   = []
		self.w 	   = []
		self.x0    = x0
		self.dt    = dt
			
	def applyInput(self, ut):
		self.u.append(ut)

		xt = self.x[-1]
		v_next      = xt[2] + self.dt * ut[0]
		theta_next  = xt[3] + self.dt * ut[1]
		avg_v       = (v_next + xt[2])/2
		avg_theta   = (theta_next + xt[3])/2
		x_next  = avg_v * self.dt * np.cos(avg_theta) + xt[0]
		y_next = avg_v * self.dt * np.sin(avg_theta) + xt[1]

		state_next = np.array([x_next, y_next, v_next, theta_next])

		self.x.append(state_next)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []

