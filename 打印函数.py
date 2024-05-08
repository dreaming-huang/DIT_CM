import numpy as np
import matplotlib.pyplot as plt

sigma_data=0.5

# 请在这里定义您的函数
def my_function(t):
    # 补全您的函数表达式
    t=t+1e-44
    return sigma_data**2/t**2+sigma_data**2
    # pass

# 生成一系列x值
x_values = np.linspace(0, 1000, 100000)  # 范围和数量可根据需要调整

# 采样函数的点
y_values = my_function(x_values)

# 绘制图表
plt.plot(x_values, y_values, label='Function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Sampling')
plt.grid(True)
plt.legend()
plt.show()
