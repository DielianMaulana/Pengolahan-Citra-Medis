import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from PIL import Image
import cv2
import pandas as pd
from scipy import ndimage as ndi

# Definisikan fungsi untuk menampilkan gambar
def show_image(image, title='Image', width=300):  # Menambahkan parameter width
    st.image(image, caption=title, use_column_width=False, width=width)  # Mengatur lebar gambar

# Definisikan fungsi untuk menampilkan histogram
def plot_histogram(image, title='Histogram'):
    plt.figure(figsize=(10, 4))
    plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

# Definisikan fungsi untuk pemrosesan gambar
def histogram_equalization(image):
    import numpy as np

    # Hitung histogram
    hist_values, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Hitung CDF
    cdf = hist_values.cumsum()
    cdf_normalized = cdf * hist_values.max() / cdf.max()  # Normalisasi CDF

    # Lakukan histogram equalization
    im_equalized = np.interp(image.flatten(), np.arange(256), cdf_normalized)
    im_equalized = im_equalized.reshape(image.shape)

    # Untuk membenarkan clipping
    im_equalized = im_equalized / np.amax(im_equalized)
    im_equalized = np.clip(im_equalized * 255, 0, 255).astype(np.uint8)

    return im_equalized

def adaptive_histogram_equalization(image):
    return exposure.equalize_adapthist(image)

def contrast_stretching(image):
    # Parameter manual dari pengguna (bisa juga diatur agar bisa dikonfigurasi lewat UI)
    c, d = 8, 150
    a, b = 0, 255

    # Konversi ke float untuk mencegah overflow
    stretched_image = (image.astype(np.float32) - c) * ((b - a) / (d - c)) + a

    # Clipping supaya nilai tetap di antara 0 - 255
    stretched_image = np.clip(stretched_image, a, b)

    return stretched_image.astype(np.uint8)

def image_negative(image):
    return 255 - image

def apply_filter(image, filter_type, kernel_size):
    image = image.astype(np.uint8)

    if filter_type == 'Gaussian Filter':
        sigma = kernel_size / 3.0  # Penyesuaian nilai sigma berdasarkan kernel size
        return ndi.gaussian_filter(image, sigma=sigma)
    elif filter_type == 'Max-Min Filter':
        output = np.zeros_like(image)

        offset = kernel_size // 2
        for j in range(offset, image.shape[0] - offset):
            for i in range(offset, image.shape[1] - offset):
                local_region = image[j - offset: j + offset + 1, i - offset: i + offset + 1]
                output[j, i] = np.max(local_region) - np.min(local_region)
        return output
    elif filter_type == 'Median Filter':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'Mean Filter':
        pad = kernel_size // 2
        padded_image = np.pad(image, pad_width=pad, mode='reflect')
        output = np.zeros_like(image)

        if kernel_size == 3:
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    region = padded_image[j:j + 3, i:i + 3]
                    output[j, i] = int(np.mean(region))

        elif kernel_size == 5:
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    region = padded_image[j:j + 5, i:i + 5]
                    output[j, i] = int(np.mean(region))

        elif kernel_size == 7:
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    region = padded_image[j:j + 7, i:i + 7]
                    output[j, i] = int(np.mean(region))

        return output.astype(np.uint8)


# Judul aplikasi
st.title("Bismilah Pengolahan Citra Medis")

# Upload file gambar
uploaded_file = st.file_uploader("Unggah Citra", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar menggunakan PIL
    image = Image.open(uploaded_file).convert('L')  # Mengonversi ke grayscale
    image = np.array(image)  # Mengonversi ke array NumPy untuk pemrosesan lebih lanjut

    st.subheader("Original Image")
    show_image(image, "Original Image", width=300)  # Mengatur lebar gambar
    plot_histogram(image, "Original Histogram")

    # Pilih teknik peningkatan citra
    enhancement_options = st.multiselect("Select Enhancement Techniques", 
                                          ["Histogram Equalization", "Adaptive Histogram Equalization", 
                                           "Contrast Stretching", "Image Negative"])

    # Menyimpan hasil untuk setiap filter
    results = []

    # Terapkan teknik peningkatan citra yang dipilih
    enhanced_images = {}
    for enhancement_name in enhancement_options:
        if enhancement_name == "Histogram Equalization":
            enhanced_images[enhancement_name] = histogram_equalization(image)
        elif enhancement_name == "Adaptive Histogram Equalization":
            enhanced_images[enhancement_name] = adaptive_histogram_equalization(image)
        elif enhancement_name == "Contrast Stretching":
            enhanced_images[enhancement_name] = contrast_stretching(image)
        elif enhancement_name == "Image Negative":
            enhanced_images[enhancement_name] = image_negative(image)

    # Tampilkan gambar hasil peningkatan citra dan histogramnya
    for enhancement_name, enhanced_image in enhanced_images.items():
        st.subheader(f"{enhancement_name} Result")
        show_image(enhanced_image, enhancement_name, width=300)  # Mengatur lebar gambar
        plot_histogram(enhanced_image, f"{enhancement_name} Histogram")

    # Pilih ukuran kernel
    kernel_sizes = [3, 5, 7]
    filter_options = ["Gaussian Filter", "Max-Min Filter", "Median Filter", "Mean Filter"]

    # Menyimpan hasil filter
    filter_results = []

    for enhancement_name, enhanced_image in enhanced_images.items():
        for kernel_size in kernel_sizes:
            for filter_name in filter_options:
                filtered_image = apply_filter(enhanced_image, filter_name, kernel_size)
                mse = mean_squared_error(image, filtered_image)
                psnr = peak_signal_noise_ratio(image, filtered_image)
                filter_results.append({
                    "Enhancement": enhancement_name,
                    "Filter": filter_name,
                    "Kernel Size": kernel_size,
                    "MSE": mse,
                    "PSNR": psnr
                })

                # Tampilkan gambar hasil filter dan histogram
                st.subheader(f"{enhancement_name} dengan {filter_name} (Kernel Size: {kernel_size})")
                show_image(filtered_image, f"{enhancement_name} dengan {filter_name} (Kernel Size: {kernel_size})", width=300)  # Mengatur lebar gambar
                plot_histogram(filtered_image, f"{enhancement_name} dengan {filter_name} Histogram")

    # Tampilkan hasil dalam bentuk tabel
    results_df = pd.DataFrame(filter_results)
    st.subheader("Tabel Hasil Evaluasi")
    st.write(results_df)

    # Menentukan filter terbaik berdasarkan MSE terkecil dan PSNR tertinggi
    best_mse_row = results_df.loc[results_df['MSE'].idxmin()]
    best_psnr_row = results_df.loc[results_df['PSNR'].idxmax()]

    # Menentukan filter terbaik berdasarkan kombinasi MSE terkecil dan PSNR tertinggi
    results_df['Score'] = results_df['PSNR'] / results_df['MSE']  # Skor gabungan
    best_combined_row = results_df.loc[results_df['Score'].idxmax()]

    st.write(f"**Filter yang paling optimal :** {best_combined_row['Filter']} dengan nilai MSE: {best_combined_row['MSE']:.2f} dan PSNR: {best_combined_row['PSNR']:.2f} dB")