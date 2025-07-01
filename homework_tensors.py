import torch
import numpy as np

# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
random_tensor = torch.rand(3, 4)
# - Тензор размером 2x3x4, заполненный нулями
zeros_tensor = torch.zeros(2, 3, 4)
# - Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones(5, 5)
# - Тензор размером 4x4 с числами от 0 до 15
tensor = torch.arange(0, 16, 1)
tensor2 = tensor.reshape(4, 4)

a = torch.rand(3, 4)
b = torch.rand(4, 3)

# - Транспонирование тензора A
print(a.T)
# - Матричное умножение A и B
print(a @ b)
# - Поэлементное умножение A и транспонированного B
print(a * b.T)
# - Вычислите сумму всех элементов тензора A
print(a.sum())

tensor5 = torch.rand(5, 5, 5)
# - Первая строка
print(tensor5[0, 0])
# - Последний столбец
print(tensor5[4, :, 4])
# - Подматрица размером 2x2 из центра тензора
print(tensor5[2, 1:3, 1:3])
# - Все элементы с четными индексами
print(tensor5[::2, ::2, ::2])

tensor24 = torch.arange(0, 24, 1)
# - 2x12
tensor2x12 = tensor24.reshape(2, 12)
# - 3x8
tensor3x8 = tensor24.reshape(3, 8)
# - 4x6
tensor4x6 = tensor24.reshape(4, 6)
# - 2x3x4
tensor2x3x4 = tensor24.reshape(2, 3, 4)
# - 2x2x2x3
tensor2x2x2x3 = tensor24.reshape(2, 2, 2, 3)
