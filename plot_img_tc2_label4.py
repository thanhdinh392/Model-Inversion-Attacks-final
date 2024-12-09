# importing libraries

import matplotlib.pyplot as plt
import numpy as np

# Các giá trị alpha tương ứng
alpha = [1.0, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 1500, 2000]

# Dữ liệu cho central
ssimloss_li_central = [0.6799, 0.5922, 0.5327, 0.4968, 0.4751, 0.4576, 0.4511, 0.4663, 0.4626, 0.4709, 0.5286, 0.56, 0.5821, 0.6514, 0.7798, 0.8012, 0.8651]
mseloss_li_central = [0.0434, 0.0394, 0.0365, 0.0355, 0.0348, 0.0335, 0.0318, 0.0314, 0.0305, 0.0306, 0.0323, 0.035, 0.0359, 0.0388, 0.0561, 0.0597, 0.0599]
pixelloss_li_central = [0.3686, 0.3305, 0.3045, 0.2849, 0.2737, 0.2685, 0.2722, 0.2845, 0.2888, 0.2903, 0.3161, 0.326, 0.3403, 0.3688, 0.4075, 0.4134, 0.4209]

# Dữ liệu cho fed
ssimloss_li_fed = [0.6853, 0.6727, 0.6524, 0.5988, 0.5678, 0.5156, 0.5042, 0.5041, 0.5099, 0.5336, 0.5564, 0.6007, 0.6422, 0.6608, 0.7008, 0.7682, 0.8339]
mseloss_li_fed = [0.0437, 0.0432, 0.0415, 0.0394, 0.0384, 0.0373, 0.0356, 0.0346, 0.0343, 0.0346, 0.0347, 0.0393, 0.0436, 0.0401, 0.0455, 0.0491, 0.0655]
pixelloss_li_fed = [0.3705, 0.3633, 0.3558, 0.3329, 0.3177, 0.2908, 0.291, 0.2926, 0.2963, 0.3112, 0.3267, 0.3258, 0.3365, 0.3605, 0.3761, 0.4327, 0.3709]

# Initialize the subplot function with shared x-axis
fig, axis = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
# SSIM
axis[0].plot(alpha, ssimloss_li_central, label='SSIM Loss (Central)', color='blue', marker='o', linestyle='-')
axis[0].plot(alpha, ssimloss_li_fed, label='SSIM Loss (Fed)', color='blue', marker='x', linestyle='--')
axis[0].set_title('Change of SSIM Loss, MSE Loss and Pixel Loss: Central vs Fed', fontdict=dict(weight='bold'))
axis[0].set_ylabel("SSIM Loss", fontdict=dict(weight='bold')).set_color('blue')
axis[0].set_yticks(np.arange(0.4, 1, 0.1))  # Set y-axis ticks for sine
axis[0].grid(True)
axis[0].legend(loc="lower right")


# Vẽ MSE Loss
axis[1].plot(alpha, mseloss_li_central, label='MSE Loss (Central)', color='red', marker='o', linestyle='-')
axis[1].plot(alpha, mseloss_li_fed, label='MSE Loss (Fed)', color='red', marker='x', linestyle='--')
axis[1].set_ylabel("MSE Loss", fontdict=dict(weight='bold')).set_color('red')
axis[1].set_yticks(np.arange(0.02, 0.08, 0.01))
axis[1].grid(True)
axis[1].legend(loc="lower right")


# Vẽ Pixel Loss
axis[2].plot(alpha, pixelloss_li_central, label='Pixel Loss (Central)', color='green', marker='o', linestyle='-')
axis[2].plot(alpha, pixelloss_li_fed, label='Pixel Loss (Fed)', color='green', marker='x', linestyle='--')
# Plot Tangent Function (subplot 3)
axis[2].set_ylabel("Pixel Loss", fontdict=dict(weight='bold')).set_color('green')
axis[2].set_yticks(np.arange(0.2, 0.5, 0.05))
axis[2].grid(True)
axis[2].legend(loc="lower right")
axis[2].set_xlabel("1/Alpha", fontdict=dict(weight='bold'))

# Adjust layout to prevent overlap
plt.tight_layout()
# Cài đặt log-scale cho trục x
plt.xscale('log')
# Saving images
plt.savefig("img_tc2_label4.jpg")
# Show plot
plt.show()
