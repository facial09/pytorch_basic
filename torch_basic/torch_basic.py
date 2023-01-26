# 2D : |t| = (batch size, dim)
# 3D : |t| = (batch size, width, height) _ Computer Vision
# 3D : |t| = (batch size, length, dim) _ NLP : dim 크기의 문장이 batch size 만큼 존재한다.

import numpy as np
import torch

# 1D Array with Numpy

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print('Rank of t : ',t.ndim)
print('Shape of t : ',t.shape)
print(t[0],t[3:5],t[:-1])

# 2D Array with Numpy

t = np.array([[1., 2., 3.,],[4., 5., 6., ],[7., 8., 9.,]])
print(t)

# 1D Array with Pytorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.,])
print(t)
print('Rank of t : ',t.dim())
print('Shape of t : ',t.shape)
print(t.size())
print(t[0],t[3:5],t[:-1])

# 2D Array with Pytorch
t = torch.FloatTensor([[1., 2., 3.,],[4., 5., 6.,]])
print(t)

# Pytorch Broadcasting

m1 = torch.FloatTensor([3, 3])
m2 = torch.FloatTensor([5, 5])
print(m1 + m2)

# Broadcasting - vector + scalar

m1 = torch.FloatTensor([3, 3]) # Vector
m2 = torch.FloatTensor([5]) # Scalar => Vector [5, 5]
print(m1 + m2)


# Broadcasting - different shape Vectors

m1 = torch.FloatTensor([1, 3]) # (1x2)
m2 = torch.FloatTensor([[4],[5]]) # (2x1)
print(m1 + m2)
print((m1+m2).shape)

# Multiplication vs Matrix Multiplication

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print(m1.matmul(m2)) # Matrix Multiplication
print(m1.mul(m2)) # Broadcasting and Hadamard product

# Mean 
t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [7, 10]])
print(t.mean())
print(t.mean(dim=0)) # 2x2 -> 1x2
print(t.mean(dim=1)) # 2x2 -> 2x1
print(t.mean(dim=-1))

# Max and Argmax

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max())
print(t.max(dim=0)) # return [Max , Argmax]
print(t.max(dim=0)[0]) # Max
print(t.max(dim=0)[1]) # Argmax

# View(Reshape)
t = np.array([[[0, 1, 2],
[3, 4, 5]],
[[6, 7, 8],
[9, 10, 11]]])

ft = torch.FloatTensor(t)
print(ft.shape)
print(ft.view([-1, 3])) # -1이 있는 자리는 자동으로 맞춰준다.
print(ft.view([-1, 3]).shape)
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# Squeeze
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
print(ft.squeeze()) # dimension의 element가 1개인경우 squeeze를 사용하면 없애줌.
print(ft.squeeze().shape)

print(ft.squeeze(dim=1)) # dim=0인 경우 변화가 없음.

# Unsqueeze
ft = torch.FloatTensor([0,1 ,2])
print(ft.unsqueeze(0)) # dim=0 위치에 1을 만들어줌.
print(ft.unsqueeze(1))

ft = torch.FloatTensor([[0, 1, 2]])
print(ft.shape)
print(ft.unsqueeze(0).shape)

# Type Casting
lt = torch.LongTensor([1, 2, 3, 4])
lt2 = lt.float()
print(lt2)

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

# concatenate
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0 ))
print(torch.cat([x, y], dim=1 ))

# Stacking

x = torch.FloatTensor([1, 4])
x2 = torch.FloatTensor([2, 5])
x3 = torch.FloatTensor([3, 6])

print(torch.stack([x, x2, x3]))
print(torch.stack([x, x2, x3], dim=1))

print(torch.cat([x.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)], dim=0))
print(torch.cat([x.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)], dim=1))

# Ones and Zeros

x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x))
print(torch.zeros_like(x)) # device도 동일한 사용이 이루어짐.

# In-Place Operation
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)