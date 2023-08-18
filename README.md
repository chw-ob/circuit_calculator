# 一个简单的电路可视化计算软件
本软件计划以Python的tkinter图形库为基础，
## 注意事项
组员提交的branch以自己名字的中文拼音手写命名，比如陈汉文提交的branch就应该是chw，尽量不要使用全局变量.需要的变量以函数变量作为输入即可。
## 规划的功能
1. 创建一个元件库（包含直流和正弦电压源,电阻,电感,电容），允许用户借助元件库自定义电路
2. 计算任意两节点间的等效电阻
3. 计算电路在直流和正弦激励下的响应


# 任务分工
## 陈汉文
1. 软件算法设计
   1. 从电路数据中获取到电路的拓扑结构
   2. 由拓扑结构计算出任意两点间的等效电阻
   3. 计算出任意电位点的响应


## 刘俊宇
1. 人机交互设计
   1. 交互逻辑实现，包括鼠标拖动元件库，相关按钮触发逻辑
   2. 获取并记录电路图的拓扑数据，以下列形式记录：

   e=[kind,position,net,property],其中e代表单个元件，kind代表他是电阻还是电容etc..，position则是他在主界面的位置
。net则是他的端口的电位点，而property是他自己的属性，比如如果king是电阻就代表property是电阻的电阻大小


## 腾跃俊
1. 软件GUI设计
   1. 元件库的图形设计
   2. 主界面UI设计以及输入元件信息后的可视化，元件信息如上图所示


# 阶段一
我们以一个简单的星形电路作为起点，各组员调试自己的功能
这样的数据就是
e1=["source_v",[0,0],[0,1],6]
e2=["r",[0,0],[1,2],6]
e3=["r",[0,0],[0,2],6]
e4=["r",[0,0],[0,2],6]
