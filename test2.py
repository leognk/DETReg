import torch
a = torch.tensor([1.], requires_grad=True)
y = torch.zeros((10))
gt = torch.zeros((10))

y[0] = a
y[1] = y[0] * 2
y.retain_grad()

loss = torch.sum((y-gt) ** 2)
loss.backward()
print(y.grad)