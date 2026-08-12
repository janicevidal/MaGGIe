import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_to_sigmoid(x):
    return (np.tanh(x) + 1) / 2

# 生成 x 轴数据
x = np.linspace(-6, 6, 500)

# 计算 y 值
y_sigmoid = sigmoid(x)
y_tanh = tanh_to_sigmoid(x)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y_sigmoid, label='sigmoid(x)', linewidth=2)
plt.plot(x, y_tanh, label='(tanh(x)+1)/2', linestyle='--', linewidth=2)

# 辅助线
plt.axhline(0, color='black', linewidth=0.5)
plt.axhline(1, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5)
plt.ylim(-0.05, 1.05)
plt.xlabel('x')
plt.ylabel('输出值')
plt.title('Sigmoid vs (tanh+1)/2 对比')
plt.legend()
plt.grid(True)
plt.savefig("test.png")