import numpy as np

def calculate_complex_response(response, theta_degrees):
    # 将角度转换为弧度
    theta_radians = np.deg2rad(theta_degrees)

    # 计算 exp(2i * theta)
    complex_exp = np.exp(2j * theta_radians)

    # 计算 response * exp(2i * theta)
    complex_response = response * complex_exp

    return complex_response

def calculate_weighted_sum(response, theta_degrees):
    # 计算复数形式的神经元响应
    complex_response = calculate_complex_response(response, theta_degrees)

    # 计算加权和
    weighted_sum = np.sum(complex_response) / np.sum(response)

    # 取模
    magnitude = np.abs(weighted_sum)

    return magnitude

# 示例数据
response = np.array([10, 20, 30, 40, 50, 60, 70, 80])
theta_degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315])

# 计算加权和的模
result = calculate_weighted_sum(response, theta_degrees)

print(f"Weighted Sum Magnitude: {result}")
