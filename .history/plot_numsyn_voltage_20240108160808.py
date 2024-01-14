import matplotlib.pyplot as plt

v_list = [-42.81389957673183,-24.79735767367017,-25.067427394556727,-21.291730788632318,
          -17.38007166494208,-16.14881725763812,-9.756035311627917,-15.288329391101405,
          -12.787297011646862,-8.247841236705508]

plt.plot(v_list)
x_values = np.linspace(-10, 10, 100)

# Calculate corresponding y values for y = x
y_values = x_values

# Plot the line with a dash linestyle
plt.plot(x_values, y_values, label='y = x', linestyle='--')
plt.show()