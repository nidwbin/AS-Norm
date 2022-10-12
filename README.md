# ASNorm
[中文文档|Chinese](README_zh.md)    

This is my implement of adaptive score normalization (AS-Norm) in speaker verification with pytorch.  

At the same time, I provide a brainless and safe tool for "cuda()", "detach()", "cpu()", "numpy()" in pytorch and some useful function in speaker verification.
## Usage
```python
import numpy

from tool import as_norm, to, setup_seed

# setup random seed
setup_seed(10)
a = numpy.random.random((10, 1))
b = numpy.random.random((10, 1000))
c = numpy.random.random((10, 1000))

# a1 should be a torch.Tensor and on gpu:0
a1 = as_norm(a, b, c, gpu=0, array=False)
print(a1)

# a2 should be a numpy.ndarray
a2 = as_norm(a, b, c, gpu=0)
print(a2)

# it should print "'NoneType' object has no attribute 'to'"
to.gpu(None, device=0, catch_except=True)

# a should be a torch.Tensor
a = to.tensor(a)
print(a)

# b and c should be a torch.Tensor
b, c = to.tensor([b, c])
print(b)
print(c)

# a, b and c should on gpu:0 now
# if there was not a gpu, a, b and c should on cpu
a, b, c = to.gpu([a, b, c], device=0)

# it should print a numpy.ndarray
print(to.array(a))

# it should be as same as to.array(a)
print(to.cpu(to.detach(a)))

# d should be a torch.Tensor with shape(2, 10, 1000)
d = to.tensor([b, c], package=True, new_dim=True)
print(d)

# e should be a torch.Tensor with shape(20, 1000)
e = to.tensor([b, c], package=True, new_dim=False)
print(e)
``` 
## Reference
```
@inproceedings{matejka2017analysis,
  title={Analysis of Score Normalization in Multilingual Speaker Recognition.},
  author={Matejka, Pavel and Novotn{\`y}, Ondrej and Plchot, Oldrich and Burget, Lukas and S{\'a}nchez, Mireia Diez and Cernock{\`y}, Jan},
  booktitle={Interspeech},
  pages={1567--1571},
  year={2017}
}
```