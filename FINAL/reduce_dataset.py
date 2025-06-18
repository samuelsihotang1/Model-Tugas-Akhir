import os
import random
import shutil

def delete_images_in_directory(directory, percentage_to_keep=0.1):
    # Daftar semua file di direktori
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Jumlah gambar yang akan dipertahankan
    num_files_to_keep = int(len(files) * percentage_to_keep)
    
    # Pilih secara acak gambar yang akan dipertahankan
    files_to_keep = random.sample(files, num_files_to_keep)
    
    # Hapus gambar yang tidak terpilih
    for file in files:
        if file not in files_to_keep:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

def process_classes(base_directory):
    # Daftar semua kelas (subdirektori) di direktori dasar
    classes = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    for class_name in classes:
        class_directory = os.path.join(base_directory, class_name)
        print(f"Processing class: {class_name}")
        delete_images_in_directory(class_directory)

# Ganti 'train' dengan path ke direktori tempat data Anda berada
base_directory = '/home/tasi2425111/restructured_resized_imagenet/train/'
process_classes(base_directory)