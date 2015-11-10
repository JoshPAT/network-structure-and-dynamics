### Code Review

#### English Version:
---

#### Improvements
- Fix the 'Graph' class in previous exericse. Make it more concise and esay reading.
- Automatically create result file based on datasets. 
- Add argparse functions to make debug and test easiler.

##### 
- 

#### Chinese Version:
---

##### 功能改进：
- 重新编写了原来的Graph类，提高了这个类的可读性和效率。
- 根据数据名称，自动生成相应的命名的文件和相关数据。
- 添加命令行控制功能，优化命令行调试过程。

##### 需改进
- 暂无

##### 心得体会
- 在TBF中采用的高效算法中对“最小链路比较值”的设置对结果造成很大的差异。
- 采用zip(*sorted(dict))来剥离排序过后的key和value。