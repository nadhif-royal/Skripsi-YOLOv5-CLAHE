import cv2
import numpy as np

def apply_clahe_lab(image_bgr):
    """
    Fungsi untuk menerapkan CLAHE pada saluran L* (Lightness)
    menggunakan parameter optimal hasil Grid Search skripsi.
    """
    # 1. Konversi dari BGR ke CIE L*a*b*
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    # 2. Ekstraksi / Isolasi saluran
    l, a, b = cv2.split(lab)
    
    # 3. Inisiasi dan eksekusi CLAHE (Berdasarkan hasil uji: Clip 3.0, Grid 4x4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    
    # 4. Duplikasi saluran L menjadi 3 channel (L*L*L*)
    # Menghindari noise dari saluran 'a' dan 'b' serta menjaga kompatibilitas input model
    l_l_l = cv2.merge((cl, cl, cl))
    
    return l_l_l