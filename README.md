# Debug
> 模型训练时候出现的Bug，以及解决方法



##1.TypeError: new(): argument 'size' must be tuple of ints, but found element of type NoneType at pos 2

> https://blog.csdn.net/stickmangod/article/details/86308982

模型初始化的时候，都没有传入维度进行初始化

![image-20220309211324124](https://gitee.com/wanghui88888888/picture/raw/master/img/image-20220309211324124.png)
