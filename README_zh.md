# ASNorm
[English|英文文档](README.md)    

这是我用pytorch实现的说话人识别自适应分数归一化(AS-Norm)仓库。

仓库里还提供了一些说话人识别和pytorch常用的安全无脑的方法。
## 用法
```python
import numpy

from tool import as_norm, to, setup_seed

# 设置随机种子
setup_seed(10)
a = numpy.random.random((10, 1))
b = numpy.random.random((10, 1000))
c = numpy.random.random((10, 1000))

# a1应该是一个在gpu:0上的torch.Tensor
a1 = as_norm(a, b, c, gpu=0, array=False)
print(a1)

# a2应该是一个numpy.ndarray
a2 = as_norm(a, b, c, gpu=0)
print(a2)

# 这里应该捕获异常并输出"'NoneType' object has no attribute 'to'"
to.gpu(None, device=0, catch_except=True)

# a现在应该是一个在cpu上的torch.Tensor
a = to.tensor(a)
print(a)

# b和c现在应该都是在cpu上的torch.Tensor
b, c = to.tensor([b, c])
print(b)
print(c)

# a、b、c现在应该全在gpu:0上
# 如果没有gpu,那么a、b、c现在应该全在cpu
a, b, c = to.gpu([a, b, c], device=0)

# 应该输出一个a numpy.ndarray
print(to.array(a))

# 输出结果应该和to.array(a)一样
print(to.cpu(to.detach(a)))

# d应该是一个在gpu:0上的torch.Tensor，并且shape为(2, 10, 1000)
d = to.tensor([b, c], package=True, new_dim=True)
print(d)

# e应该是一个在gpu:0上的torch.Tensor，并且shape为(20, 1000)
e = to.tensor([b, c], package=True, new_dim=False)
print(e)
``` 
## 参考文献
```
@inproceedings{matejka2017analysis,
  title={Analysis of Score Normalization in Multilingual Speaker Recognition.},
  author={Matejka, Pavel and Novotn{\`y}, Ondrej and Plchot, Oldrich and Burget, Lukas and S{\'a}nchez, Mireia Diez and Cernock{\`y}, Jan},
  booktitle={Interspeech},
  pages={1567--1571},
  year={2017}
}
```