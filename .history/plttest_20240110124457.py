import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2 * x)
y4 = np.cos(2 * x)

# 直接使用 plt.plot 画多个曲线
plt.plot(x, y1, label='Curve 1')
plt.plot(x, y2, label='Curve 2')
plt.plot(x, y3, label='Curve 3')
plt.plot(x, y4, label='Curve 4')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('Multiple Curves with plt.plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()
