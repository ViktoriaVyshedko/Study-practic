import torch

# Тензоры x, y, z с requires_grad=True
x = torch.rand(2, requires_grad=True)
y = torch.rand(2, requires_grad=True)
z = torch.rand(2, requires_grad=True)

print(x, y, z)

# Функция: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = (x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z).sum()

f.backward()
# Градиенты по всем переменным
print(x.grad, y.grad, z.grad)
# по x = 2 * x + 2 * y * z = (2 * 0,6307 + 2 * 0.4167 * 0.2838, 2 * 0,4949 + 2 * 0.8166 * 0.1845) = (1,4979, 1.2911)
# по y = 2 * y + 2 * x * z = (2 * 0,4167 + 2 * 0.6307 * 0.2838, 2 * 0,8166 + 2 * 0.4949 * 0.1845) = (1,1914, 1.8158)
# по y = 2 * z + 2 * x * y = (2 * 0,2838 + 2 * 0.6307 * 0.4167, 2 * 0,1845 + 2 * 0.4949 * 0.8166) = (1,0932, 1.1773)

x = torch.rand(2, requires_grad=True)
y_true = torch.rand(2, requires_grad=True)

w = torch.rand(2, requires_grad=True)
b = torch.rand(2, requires_grad=True)
y_pred = w * x + b

n = x.shape[0]

# Функция MSE:
MSE = ((y_pred - y_true) ** 2).sum() / n

MSE.backward()
# Градиенты по w и b    
print(w.grad, b.grad)

x = torch.rand(2, requires_grad=True)

print(x)

# Cоставная функция: f(x) = sin(x^2 + 1)
fx = ((x ** 2 + 1).sin()).sum()

# Градиент df/dx
print(torch.autograd.grad(fx, x))
