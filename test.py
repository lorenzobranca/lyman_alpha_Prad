from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import deepxde as dde
from scipy.integrate import quad
from deepxde.backend import tf
import matplotlib.pyplot as plt

def stupid_integral(x,z):
    
    result = tf.math.reduce_sum((z[1:]+z[0:-1])/2.*(x[1:]-x[0:-1]))
    print('!!!!!!!!!!!', result)
    return result

def integrand(x,z):
    return z

def test(x, vec):

    y, z = vec[:,0:1], vec[:,1:2] 

    dy_dx = dde.grad.jacobian(vec, x, i=0, j=0)
    dz_dx = dde.grad.jacobian(vec, x, i=1, j=0)
    
    k = stupid_integral(x,z)

    eq1 = dy_dx - x - k#quad(integrand,0, 1, args = (z))[0]
    eq2 = dz_dx + x
    

    return [eq1, eq2]

def test_v2(x, vec):

    y, z = vec[:,0:1], vec[:,1:2]

    dy_dx = dde.grad.jacobian(vec, x, i=0, j=0)
    dz_dx = dde.grad.jacobian(vec, x, i=1, j=0)

    k = stupid_integral(x,z)

    eq1 = dy_dx - x - k#quad(integrand,0, 1, args = (z))[0]
    eq2 = dz_dx + x


    return [eq1, eq2]


def analytical_sol(x):

    sol1 = x*x/2+11./6*x+1
    sol2 = -x*2/2.+2
    return sol1, sol2

def initial(_, on_initial):
        return on_initial

def boundary(_, on_boundary):
    return on_boundary

def fun_ICy(y):
    return 1.

def fun_ICz(z):
    return 2.

def fun_ICk(k):
    return 0

#geom = dde.geometry.geometry_2d.Rectangle((0, 1e0), (0, 1))
geom = dde.geometry.TimeDomain(0, 1)


icy = dde.IC(geom, fun_ICy, initial, component=0)
icz = dde.IC(geom, fun_ICz, initial, component=1)

data = dde.data.TimePDE(geom, test, [icy, icz], 1024, 2, num_test=528)

layer_size = [1] + [64] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

losshistory, train_state = model.train(epochs=5000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save('model_ode/model',verbose=1)

model.restore('model_ode/model-5000.ckpt',verbose=1)

x = np.linspace(0,1).reshape((-1,1))

ypred = model.predict(x)

sol1, sol2 = analytical_sol(np.linspace(0,1))

plt.plot(np.linspace(0,1),sol1)
plt.plot(np.linspace(0,1),sol2)
plt.plot(np.linspace(0,1),ypred[:,0])
plt.plot(np.linspace(0,1),ypred[:,0])

plt.show()
