import matplotlib.pyplot as plt

def two_img(f,s):
    fig = plt.figure(figsize=(10, 5)) 
    gs = gridspec.GridSpec(1, 2) 
    ax0 = plt.subplot(gs[0])
    ax0.imshow(f) ax1 = plt.subplot(gs[1])
    ax1.imshow(s)
    plt.show()
