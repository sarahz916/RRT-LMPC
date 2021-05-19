"""
Cubic spline planner
Author: Atsushi Sakai(@Atsushi_twi)

Adapted by Aaron
"""
import math
import numpy as np
import bisect
import pdb

class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    # Compute and store for a fixed set of s values along the path
    # the corresponding s, x(s), y(s), x'(s), y'(s) values for later use
    # Store points at intervals of ds path length
    def __init__(self, x, y, ds=0.05):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)
        self.ds = ds
        self.end = np.max(self.s)
        self.PointTangent = np.array([[ds*i, self.calc_position(ds*i)[0], self.calc_position(ds*i)[1], self.sx.calcd(ds*i), self.sy.calcd(ds*i)] for i in range(0,int(self.end/ds))])
        
    # Gets the total path length s
    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    # Given a particular s, find x(s), y(s)
    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    # Given s, compute gamma(s)
    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw
    
    # Given s, compute d/ds gamma(s)
    def calc_yawPrime(self, s):
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        yawPrime = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**2)
        return yawPrime
    
    # Compute the unit normal and non-unit tangent vector at given point
    def calcTangentNormal(self, s):
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        r = np.sqrt(dx**2 + dy**2)
        nx = - dy / r
        ny = dx / r
        # Return tangent, normal
        return np.array([dx, dy]), np.array([nx, ny])
        
    # Convert from s, y to x1, y1
    def calcXY(self, s, y):
        x0, y0 = self.calc_position(s)
        
        # Compute normal vector
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        r = np.sqrt(dx**2 + dy**2)
        nx = - dy / r
        ny = dx / r
        
        x1 = x0 + y * nx
        y1 = y0 + y * ny
        
        return x1, y1
        
    # Convert from x1, y1 to s, y
    def calcSY(self, x1, y1, eps = 1e-4, maxIter = 0):
        # maxIter = 10, shrink=0.9, startSize = 5
        # Compute the point which minimizes distance from x1, y1 to x0, y0 
        # Potentially refine by repeating process for increasingly
        # narrow windows or by making inner product between point - center
        # and tangent 0
        point = np.array([x1, y1])
        sVals = self.PointTangent[:,0]
        centers = self.PointTangent[:,1:3]
        dists = np.sqrt(np.sum(np.square(point - centers), axis=1))
        ind = np.argmin(dists)
        s = sVals[ind]
        # Make it positive displacement if aligned with normal and negative
        # else
        
        # Compute normal vector
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        r = np.sqrt(dx**2 + dy**2)
        nx = - dy / r
        ny = dx / r
        
        # Should be equivalent to just doing np.dot and not using dists[ind]
        y = dists[ind] * np.sign(np.dot(np.array([nx, ny]), point-centers[ind])) 
                
        # count = 0
        # while count < maxIter:
        #     dists = np.sqrt(np.sum(np.square(point - centers), axis=1))
        #     ind = np.argmin(dists)
        #     s = sVals[ind]
        #     y = dists[ind]
            
        #     pdb.set_trace()
            
        #     windowSize = shrink**count * min(self.end/2, startSize)
        #     PointTangent = np.array([[val, self.calc_position(val)[0], self.calc_position(val)[1], self.sx.calcd(val), self.sy.calcd(val)] for val in np.linspace(s-windowSize/2, s+windowSize/2)])
        #     sVals = PointTangent[:,0]
        #     centers = PointTangent[:,1:3]
        #     count += 1
        #     print('count', count)
        #     print('s', s)
        #     print('y', y)
        #     print('x, y',self.calcXY(s, y))
            
        count = 0
        inner = 1
        while abs(inner) > eps and count < maxIter:
            pdb.set_trace()
            estNormal = point - np.array(self.calc_position(s))
            tangent = [self.sx.calcd(s), self.sy.calcd(s)]
            # Use normalized inner product
            inner = np.dot(estNormal , tangent) / np.linalg.norm(estNormal) / np.linalg.norm(tangent)
            print('normal mag = ', np.linalg.norm(estNormal))
            print('tangent mag = ', np.linalg.norm(tangent))
            print('count = ', count)
            print('s = ', s)
            print('x,y = ', self.calcXY(s,y))
            print('inner = ', inner)
            s += inner # Move in direction of inner product
            y = np.linalg.norm(point - np.array(self.calc_position(s)))
            count += 1
            
        return s, y

def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s

# Given the state: s,y,v,theta and inputs: acc, theta_dot
# compute the linearized dynamics of system
def computeDynamics(s, y, v, theta, acc, theta_dot):
    pass

def main():  # pragma: no cover
    print("Spline 2D test")
    import matplotlib.pyplot as plt
    # x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    # y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    
    path = np.load('path.npy')
    
    x = path[:,0]
    y = path[:,1]
    
    ds = 0.1  # [m] distance of each interpolated points

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    
    # Plot the trajectory with dx, dy overlaid as quiver field
    # sVals = np.linspace(0, sp.end, endpoint=False)
    # xVals = [sp.calc_position(s)[0] for s in sVals]
    # yVals = [sp.calc_position(s)[1] for s in sVals]
    # dxVals = [sp.sx.calcd(s) for s in sVals]
    # dyVals = [sp.sy.calcd(s) for s in sVals]
    # plt.quiver(xVals, yVals, dxVals, dyVals, scale=1)
    
    plt.legend()

    plt.subplots(1)
    plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()

    plt.subplots(1)
    plt.title('Plotting the numerical derivative components')
    sVals = np.linspace(0, sp.end, int(sp.end/sp.ds), endpoint=False)
    xVals = [sp.calc_position(s)[0] for s in sVals]
    yVals = [sp.calc_position(s)[1] for s in sVals]
    dxVals = [sp.sx.calcd(s) for s in sVals]
    dyVals = [sp.sy.calcd(s) for s in sVals]
    plt.scatter(sVals, xVals, label='dx')
    plt.scatter(sVals, yVals, label='dy')
    plt.xlabel('s')
    plt.legend()

    # plt.subplots(1)
    # plt.title('Plotting gammaPrime(s)')
    # gammaVals = [sp.calc_yaw(s) for s in sVals]
    # gammaPrimeVals = [sp.calc_yawPrime(s) for s in sVals]
    # plt.plot(sVals[:-1], np.diff(gammaVals) / (sVals[1] - sVals[0]), label='1st Order Diff Approx')
    # plt.plot(sVals, gammaPrimeVals, label='Gamma Prime Estimate')
    # plt.legend()
    
    a = np.random.uniform(0, 5)
    b = np.random.uniform(0, 5)
    print('a',a,'b',b)
    print(sp.calcSY(*sp.calcXY(a,b))) # Has trouble
    print(sp.calcXY(*sp.calcSY(a,b))) # Works well
    pdb.set_trace()

if __name__ == '__main__':
    main()