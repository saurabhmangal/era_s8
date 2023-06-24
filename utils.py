import matplotlib.pyplot as plt

def plot_images (batch_data, batch_label, total_number_of_images, grid_row, type_of_data ):
    fig = plt.figure()
    for i in range(total_number_of_images):
        plt.subplot(grid_row,int(total_number_of_images/grid_row),i+1)
        plt.tight_layout()
        if (type_of_data =='CIFAR10'):
            plt.imshow(batch_data[i].permute(1,2,0))
        
        else:
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        
        plt.title(batch_label[i])
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        #plt.close()
    return fig
    
