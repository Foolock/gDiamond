import matplotlib.pyplot as plt

'''
# mix mapping ver2

# Data from the table
problem_size = [4.14, 6.97, 12.63, 23.96, 46.62, 91.92]
performance_naive = [2800.47, 2783.46, 2671.91, 2523.95, 2505.16, 2487.67]
performance_mix = [2852.59, 2855.59, 2817.92, 2738.77, 2773.26, 2788.98]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(problem_size, performance_naive, marker='o', label='Naive')
plt.plot(problem_size, performance_mix, marker='s', label='Mix mapping')

plt.title('Performance vs Problem Size')
plt.xlabel('Problem Size (Mpixels)')
plt.ylabel('Performance (Mpixels/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

'''

# mix mapping ver3 

# Data from the table
problem_size = [4.16, 8.32, 16.64, 33.28, 66.56]
performance_naive = [2758.87, 2747.95, 2558.04, 2541.44, 2501.17]
performance_mix = [3526.11, 3530.64, 3512.09, 3457.14, 3254.67]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(problem_size, performance_naive, marker='o', label='Naive')
plt.plot(problem_size, performance_mix, marker='s', label='Mix mapping')

plt.title('Performance vs Problem Size')
plt.xlabel('Problem Size (Mpixels)')
plt.ylabel('Performance (Mpixels/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()










