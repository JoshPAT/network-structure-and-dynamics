### Code Review


Supervised learning
监督式学习

#### English Version:
---

##### Algorithm



#### Improvements
- Apply numpy library, optimatize the perfermance.
- Automatically create result file based on the function calls it. 
    It is realized by the @decorator.
- Add argparse functions to make debug and test easiler.

##### Need to Improve
- 

#### Chinese Version:
---

##### 功能改进：
- 采用numpy内的library， 优化程序性能。
- 自动生成以函数名称命名的文件，存入相应文件夹。 动态索引由@装饰器实现
- 添加命令行控制功能，优化命令行调试过程。

##### 需改进
- 改进项目结构，采用扁平化架构模式。
- 引用Cython库，提高python性能。(Maybe)

##### 心得体会
- 如果获取数据变量的计算量不大，该数据变量可设为全局变量, 也可重新计算。
- 使用sum(x for x in list) 代替 x += x，可以更快。