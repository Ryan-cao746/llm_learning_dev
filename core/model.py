# 做一个两个隐藏层的多层感知器

import torch

# 在 PyTorch 中实现神经网络时，我们通常会继承 torch.nn.Module 类来定义我们自己的自定义网络架构。
# 这个 Module 基类提供了许多功能，使得构建和训练模型更加容易。例如，它允许我们封装层和操作，并跟踪模型的参数。
class NeuralNetwork(torch.nn.Module):

    # 将输入和输出的数量编码为变量很有用，这样可以为具有不同特征和类别数量的数据集重用相同的代码。
    def __init__(self, num_inputs, num_outputs):
        #我们在 __init__ 构造函数中定义网络层，并在 forward 方法中指定它们如何交互。
        # forward 方法描述了输入数据如何通过网络并组合成一个计算图。
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一隐藏层
            # Linear 层将输入和输出节点的数量作为参数。
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),    # 非线性激活函数被放置在隐藏层之间

            # 第二隐藏层
            # 一个隐藏层的输出节点数必须与下一个隐藏层的输入节点数相匹配。
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # forward 方法描述了输入数据如何通过网络并组合成一个计算图。
        # 把输入 x 送入 self.layers 这个网络层/层容器，做一次前向传播，得到输出 logits。x是输入张量
        logits = self.layers(x)
        return logits

    # backward 方法（我们通常不需要自己实现）在训练期间用于计算损失函数相对于模型参数的梯度
    # backward方法根据前向传播时记录下来的计算图，从 loss 出发反向计算梯度，并把梯度存到各个参数的 .grad 里。
    # PyTorch 的 autograd 会在前向传播时记录所有操作，构建一张计算图。