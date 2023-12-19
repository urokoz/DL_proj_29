import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# List of lists
train_ELBO, train_MSE, train_KLdiv = [], [], []
val_ELBO, val_MSE, val_KLdiv = [], [], []

# List of files to process
files = [f'results/vae_beta_1e-0{i}.txt' for i in range(7)]

pattern = r'train ELBO: ([\d.]+)\s+MSE: ([\d.]+)\s+KLdiv: ([\d.]+)\n' + \
          r'val\s+ELBO: ([\d.]+)\s+MSE: ([\d.]+)\s+KLdiv: ([\d.]+)'

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    matches = re.findall(pattern, content, re.MULTILINE)
    train_ELBO_file, train_MSE_file, train_KLdiv_file = [], [], []
    val_ELBO_file, val_MSE_file, val_KLdiv_file = [], [], []

    for match in matches:
        train_ELBO_file.append(float(match[0]))
        train_MSE_file.append(float(match[1]))
        train_KLdiv_file.append(float(match[2]))
        val_ELBO_file.append(float(match[3]))
        val_MSE_file.append(float(match[4]))
        val_KLdiv_file.append(float(match[5]))

    train_ELBO.append(train_ELBO_file)
    train_MSE.append(train_MSE_file)
    train_KLdiv.append(train_KLdiv_file)
    val_ELBO.append(val_ELBO_file)
    val_MSE.append(val_MSE_file)
    val_KLdiv.append(val_KLdiv_file)

# Print the shape of the lists to verify the data extraction
print("Shapes of the lists:")
print("train_ELBO:", [len(row) for row in train_ELBO])
print("train_MSE:", [len(row) for row in train_MSE])
print("train_KLdiv:", [len(row) for row in train_KLdiv])
print("val_ELBO:", [len(row) for row in val_ELBO])
print("val_MSE:", [len(row) for row in val_MSE])
print("val_KLdiv:", [len(row) for row in val_KLdiv])

betas = ['1e-00', '1e-01', '1e-02', '1e-03', '1e-04', '1e-05', '1e-06']

# Create a figure with specified size
plt.rcParams.update({'font.size': 16})

####################################################################################

fig = plt.figure(figsize=(10, 10)) 

gs = gridspec.GridSpec(2, 2, width_ratios=[7, 3], wspace=0)  

# Upper Left Subplot 
ax1 = plt.subplot(gs[0, 0])
for i, data in enumerate(train_ELBO):
    ax1.plot(data[:26], label=f'Beta: {betas[i]}')
ax1.set_title('Training ELBO')
ax1.set_ylabel('Magnitude')
ax1.grid('both')
ax1.legend(loc = 'upper center')

# Upper Right Subplot
ax2 = plt.subplot(gs[0, 1]) 
ax2_right = ax2.twinx()
for i, data in enumerate(train_ELBO):
    ax2.plot(np.arange(175, 200), data[-25:], label=f'Beta: {betas[i]}')
ax2.set_title('(last epochs)')
ax2.set_yticklabels([])
min_val, max_val = ax2.get_ylim()
ax2_right.set_ylim(min_val, max_val)  
ax2.grid('both')

# Lower Left Subplot
ax3 = plt.subplot(gs[1, 0])
for i, data in enumerate(val_ELBO):
    ax3.plot(data[:26], label=f'Beta: {betas[i]}')
ax3.set_title('Validation ELBO')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Magnitude')
ax3.grid('both')
ax3.legend(loc = 'upper center')

# Lower Right Subplot
ax4 = plt.subplot(gs[1, 1])
ax4_right = ax4.twinx()  
for i, data in enumerate(val_ELBO):
    ax4.plot(np.arange(175, 200), data[-25:], label=f'Beta: {betas[i]}')
ax4.set_title('(last epochs)')
ax4.set_xlabel('Epochs')
ax4.set_yticklabels([])
min_val, max_val = ax4.get_ylim()
ax4_right.set_ylim(min_val, max_val)  
ax4.grid('both')

# Adjusting layout and displaying the plot
plt.show()

####################################################################################
fig = plt.figure(figsize=(10, 10)) 

gs = gridspec.GridSpec(2, 2, width_ratios=[7, 3], wspace=0)  

# Upper Left Subplot
ax1 = plt.subplot(gs[0, 0])
for i, data in enumerate(train_MSE):
    ax1.plot(data[:26], label=f'Beta: {betas[i]}')
ax1.set_title('Training MSE')
ax1.set_ylabel('Magnitude')
ax1.set_ylim(0, 1)  
ax1.grid('both')
ax1.legend(loc = 'upper center')

# Upper Right Subplot
ax2 = plt.subplot(gs[0, 1]) 
ax2_right = ax2.twinx()
for i, data in enumerate(train_MSE):
    ax2.plot(np.arange(175, 200), data[-25:], label=f'Beta: {betas[i]}')
ax2.set_title('(last epochs)')
ax2.set_yticklabels([])
min_val, max_val = ax2.get_ylim()
ax2_right.set_ylim(min_val, max_val)  
ax2.grid('both')

# Lower Left Subplot
ax3 = plt.subplot(gs[1, 0])
for i, data in enumerate(train_KLdiv):
    ax3.plot(data[:26], label=f'Beta: {betas[i]}')
ax3.set_title('Training KLdiv')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Magnitude')
ax3.grid('both')
ax3.legend(loc = 'upper center')

# Lower Right Subplot
ax4 = plt.subplot(gs[1, 1])
ax4_right = ax4.twinx()  
for i, data in enumerate(train_KLdiv):
    ax4.plot(np.arange(175, 200), data[-25:], label=f'Beta: {betas[i]}')
ax4.set_title('(last epochs)')
ax4.set_xlabel('Epochs')
ax4.set_yticklabels([])
min_val, max_val = ax4.get_ylim()
ax4_right.set_ylim(min_val, max_val)  
ax4.grid('both')

plt.show()

