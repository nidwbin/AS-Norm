# ASNorm
This is my implement of adaptive score normalization (AS-Norm) in speaker verification.
At the same time, I provide a brainless and safe tool for "cuda()", "detach()", "cpu()", "numpy()" in pytorch and some useful function in speaker verification.
# Usage
```python
import numpy

from tool import as_norm, to, setup_seed

# setup random seed
setup_seed(10)
a = numpy.random.random((10, 1))
b = numpy.random.random((10, 1000))
c = numpy.random.random((10, 1000))
a1 = as_norm(a, b, c, gpu=0, array=False)
# a1 should be a torch.Tensor and on gpu:0
print(a1)

a2 = as_norm(a, b, c, gpu=0)
# a2 should be a numpy.ndarray
print(a2)

to.gpu(None, device=0, catch_except=True)
# It should print "'NoneType' object has no attribute 'to'"
a = to.tensor(a)
# a should be a torch.Tensor
print(a)

b, c = to.tensor([b, c])
# b and c should be a torch.Tensor
print(b)
print(c)

# a, b and c should on gpu:0 now
# if there was not a gpu, a, b and c should on cpu
a, b, c = to.gpu([a, b, c], device=0)
# It should print a numpy.ndarray
print(to.array(a))

# It should be as same as to.array(a)
print(to.cpu(to.detach(a)))

d = to.tensor([b, c], package=True, new_dim=True)
# d should be a torch.Tensor with shape(2, 10, 1000)
print(d)

e = to.tensor([b, c], package=True, new_dim=False)
# e should be a torch.Tensor with shape(20, 1000)
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