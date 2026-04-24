import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

def show_reconstructions(model, ds, n=8, is_vae=False):
    for x_batch, _ in ds.take(1):
        x_batch = x_batch[:n]
        break
        
    if is_vae:
        _, _, z = model.encoder(x_batch)
        x_recon = model.decoder(z).numpy()
    else:
        x_recon = model.predict(x_batch, verbose=0)
        
    x_batch = x_batch.numpy()
    mse_vals = np.mean((x_batch - x_recon) ** 2, axis=(1, 2, 3))

    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7))
    for i in range(n):
        axes[0, i].imshow(x_batch[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(x_recon[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'MSE={mse_vals[i]:.4f}', fontsize=7)
        axes[1, i].axis('off')
        
        err = np.abs(x_batch[i] - x_recon[i]).squeeze()
        axes[2, i].imshow(err, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()

# FIXED: We now pass the 'encoder' directly instead of the full model
def plot_tsne(encoder, ds_eval, class_names, is_vae=False):
    all_z, all_y = [], []
    for x, y in ds_eval.take(20): 
        if is_vae:
            z, _, _ = encoder(x) 
        else:
            z = encoder(x)
        all_z.extend(z.numpy())
        all_y.extend(y.numpy())
        
    all_z = np.array(all_z)
    all_y = np.array(all_y)
    
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(all_z)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_y, cmap='tab10', alpha=0.6, s=10)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.title("Latent Space Visualization (t-SNE)")
    plt.axis('off')
    plt.show()

def generate_samples(vae_decoder, latent_dim=64, n=32):
    random_latent_vectors = tf.random.normal(shape=(n, latent_dim))
    generated_images = vae_decoder(random_latent_vectors).numpy()
    
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    fig.suptitle('VAE — Generated Samples from Prior z ~ N(0, I)', fontsize=13)
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
