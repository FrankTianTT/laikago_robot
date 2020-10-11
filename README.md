# laikago_robot

## 组成

项目有4个模块，分别是
- robot:对机械特性和电气特性的仿真
- transfer:从机器人到Gym环境的数据处理
- builder:Gym环境的封装
- agent:深度强化学习算法的学习

下面详细介绍这四个模块

### robot

在这个模块中，我们使用python程序尽可能准确的仿真一个laikago（尽管在真实环境中板载MCU运行C++代码），这个模块的任务是让后三个模块可以（在程序层面）在真实环境和仿真环境没有区别的运行。

为了我们的建模尽可能准确，这一层使用了两种方法使得仿真环境中训练出来的agent可以运行在真实环境中：
- system identification:即尽可能完善laikago的urdf文件，以及对系统**延迟时间**的建模
- randomization：即对机械特性和电气特性的随机扰动，以及从外部施加的随机的力（尽管有资料证明两者等价）

同时，我们在这一层使用torque控制，而不使用position控制，因为我们需要自己改进PD算法，unitree公司对底层做了封装，使我们无法修改。
通过position计算torque是transfer的任务之一。
