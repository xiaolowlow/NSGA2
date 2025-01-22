# %load_ext autoreload
%autoreload 2

from nsga2 import Model


model = Model()

x = model.addVars(2, 3, name='x')
y = model.addVars(5, name='y')

def objs_func(x, y):
    return [y[0], x[0, 0]]


model.setObjective(objs_func)

model.evaluate()