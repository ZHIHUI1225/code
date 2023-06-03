import matplotlib.pyplot as plt

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Plot some data on the first subplot
ax1.plot([1, 2, 3], [4, 5, 6])
ax1.set_xlim(0,10)
ax1.set_ylim(0,10)
# Plot some data on the second subplot
ax2.plot([1, 2, 3], [6, 5, 4])

# Show the plot
plt.show()