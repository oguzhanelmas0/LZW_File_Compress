import os   #dosya yolu işlemleri kaydet falan
import math # matematiksel işlemler
import pickle #lzw dosyasını kaydetmek için
import struct #binary dosya işlemleri için
import numpy as np #görüntü işlemleri için
from PIL import Image, ImageTk #görüntüleri açmak ve göstermek için
import tkinter as tk #gui için
from tkinter import filedialog, ttk, messagebox #dosya seçme ve mesaj kutuları için



# A   LZW COMPRESSION


        # LZW SIKIŞTIRMA BYTE İÇİN
def compress_bytes(data): #veriyi sıkıştırma
    """LZW compression for byte array (image data)."""
    dict_size = 256 # 0-255 arası değerler için sözlük
    dictionary = {bytes([i]): i for i in range(dict_size)} #sözlük oluşturuldu
    result = [] #sıkıştırılmış veri
    w = bytes([data[0]]) #ilk veri alındı
    
         #Ana Sıkıştırma Algoritması
    for c in data[1:]: #verinin geri kalanı için
        c_byte = bytes([c]) #byte'a çevrildi
        wc = w + c_byte 

           #Sözlükte kontrol ve yen giriş
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            
            if dict_size >= 65536:
                dict_size = 256
                dictionary = {bytes([i]): i for i in range(dict_size)}
            w = c_byte




            #son elemanı .lzw ekleme
    if w:
        result.append(dictionary[w])
    return result



#LZW DECOMPRESSION

def decompress_bytes(compressed_data):
    """LZW decompression for byte array (image data)."""
    if not compressed_data:
        return []
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    result = bytearray()
    w = dictionary[compressed_data[0]]
    result.extend(w)
    
    
         #her kodu sözlükte kontrol eder ve yeni giriş yapar
    for k in compressed_data[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + bytes([w[0]])
        else:
            raise ValueError("Invalid compressed data: %s" % k)
        result.extend(entry)
        dictionary[dict_size] = w + bytes([entry[0]])
        dict_size += 1
        if dict_size >= 65536:
            dict_size = 256
            dictionary = {i: bytes([i]) for i in range(dict_size)}
        w = entry
    return list(result)



# Grayscale görüntüde yatay yönde piksel farkı hesaplanıyor.
#Fark bilgisi kullanılarak daha verimli sıkıştırma sağlanıyor.


def compute_difference_image(gray_array):
    """Compute horizontal difference for grayscale or single-channel array."""
    diff_array = gray_array.copy()
    diff_array[:, 1:] = gray_array[:, 1:] - gray_array[:, :-1]
    return diff_array



    #diff compress görüntüyü orjinal haline getirme
def reconstruct_from_difference(diff_array):
    """Reconstruct original array from difference array."""
    reconstructed = diff_array.copy()
    for j in range(1, diff_array.shape[1]):
        reconstructed[:, j] = reconstructed[:, j-1] + diff_array[:, j]
    return reconstructed




     #numpy array'i PIL görüntüsüne dönüştürme
def np_to_pil(array, mode='L'):
    from PIL import Image
    return Image.fromarray(np.uint8(array), mode)


     #entropy hesaplama
def compute_entropy(data):
    """Compute Shannon entropy of a 1D list of values (pixels or chars)."""
    if not data:
        return 0
    freq = {}
    for item in data:
        freq[item] = freq.get(item, 0) + 1
    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy




  #kodların ortalamasını hesaplama
def compute_average_code_length(codes):
    """Approximate average code length in bits."""
    if not codes:
        return 0.0
    max_code = max(codes)
    codelen = math.ceil(math.log2(max_code + 1)) if max_code > 0 else 1
    total_bits = codelen * len(codes)
    return total_bits / len(codes)



# .lzw dosyayı kaydetme

def save_compressed_file(file_path, compressed_data, image_size, is_grayscale=False, method="standard"):
    """Save compressed data into a .lzw file."""
    with open(file_path, 'wb') as f:
        f.write(struct.pack('?', is_grayscale))
        method_flag = 0 if method == "standard" else 1
        f.write(struct.pack('B', method_flag))
        f.write(struct.pack('<II', *image_size))
        pickle.dump(compressed_data, f)



# .lzw dosyayı yükleme

def load_compressed_file(file_path):
    """Load .lzw file, return (codes, image_size, is_grayscale, method_flag)."""
    with open(file_path, 'rb') as f:
        is_grayscale = struct.unpack('?', f.read(1))[0]
        method_flag = struct.unpack('B', f.read(1))[0]
        width, height = struct.unpack('<II', f.read(8))
        image_size = (width, height)
        compressed_data = pickle.load(f)
        return compressed_data, image_size, is_grayscale, method_flag


        #.bmp dosyasının detayları
def get_image_file_details(file_path):
    """Get details about an uncompressed image file."""
    import os
    from PIL import Image
    try:
        file_size = os.path.getsize(file_path)
        img = Image.open(file_path)
        width, height = img.size
        mode = img.mode
        
        if mode == "RGB":
            bytes_per_pixel = 3
        elif mode == "RGBA":
            bytes_per_pixel = 4
        elif mode == "L":
            bytes_per_pixel = 1
        else:
            bytes_per_pixel = 0
        
        pixel_count = width * height
        raw_data_size = pixel_count * bytes_per_pixel
        return {
            "file_name": os.path.basename(file_path),
            "file_size": file_size,
            "dimensions": f"{width}x{height}",
            "mode": mode,
            "pixel_count": pixel_count,
            "bytes_per_pixel": bytes_per_pixel,
            "raw_data_size": raw_data_size,
            "file_extension": os.path.splitext(file_path)[1].lower(),
        }
    except Exception as e:
        messagebox.showerror("Error", f"Failed to get image details: {str(e)}")
        return None
    




    #.lzw dosyasının detayları

def get_lzw_file_details(file_path):
    """Get details about a compressed LZW file."""
    import os
    try:
        file_size = os.path.getsize(file_path)
        codes, image_size, is_grayscale, method_flag = load_compressed_file(file_path)
        
        width, height = image_size
        if is_grayscale:
            bytes_per_pixel = 1
            mode = "L (Grayscale)"
        else:
            bytes_per_pixel = 3
            mode = "RGB (Color)"
        
        pixel_count = width * height
        raw_data_size = pixel_count * bytes_per_pixel
        compression_method = "Difference-based" if method_flag == 1 else "Standard"
        compression_ratio = raw_data_size / file_size if file_size > 0 else 0
        
        return {
            "file_name": os.path.basename(file_path),
            "file_size": file_size,
            "dimensions": f"{width}x{height}",
            "mode": mode,
            "pixel_count": pixel_count,
            "bytes_per_pixel": bytes_per_pixel,
            "raw_data_size": raw_data_size,
            "compression_method": compression_method,
            "compression_ratio": compression_ratio,
            "total_codes": len(codes),
            "file_extension": ".lzw",
            "codes_list": codes
        }
    except Exception as e:
        messagebox.showerror("Error", f"Failed to get LZW details: {str(e)}")
        return None




# B   GUI	
class LZWImageCompressorApp:
    
    
         #GUI OLUŞTURMA
    def __init__(self, root):
        self.root = root
        self.root.title("LZW Image Compression (GUI)")
        self.root.geometry("1000x700")
        
        

        self.original_image = None
        self.original_image_path = None
        self.decompressed_image = None
        self.lzw_file_path = None
        self.is_grayscale = tk.BooleanVar(value=False)

        

        self.last_entropy = None
        self.last_avg_code_len = None
        
        self.create_widgets()

    def create_widgets(self):
        

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        

        top_frame = ttk.Frame(self.main_frame, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        left_col = ttk.Frame(top_frame)
        left_col.pack(side=tk.LEFT, padx=5)

        self.btn_open_image = ttk.Button(left_col, text="Open Image", command=self.open_image)
        self.btn_open_image.pack(pady=5, fill=tk.X)

        self.chk_grayscale = ttk.Checkbutton(left_col, text="Grayscale Compression", variable=self.is_grayscale)
        self.chk_grayscale.pack(pady=5, fill=tk.X)

        self.btn_compress = ttk.Button(left_col, text="Compress", command=self.compress_image)
        self.btn_compress.pack(pady=5, fill=tk.X)
        
        center_col = ttk.Frame(top_frame)
        center_col.pack(side=tk.LEFT, padx=5)

        self.btn_load_compressed = ttk.Button(center_col, text="Load Compressed", command=self.load_compressed)
        self.btn_load_compressed.pack(pady=5, fill=tk.X)

        self.btn_decompress = ttk.Button(center_col, text="Decompress", command=self.decompress_to_bmp)
        self.btn_decompress.pack(pady=5, fill=tk.X)
        
        right_col = ttk.Frame(top_frame)
        right_col.pack(side=tk.LEFT, padx=5)

        self.btn_gray_preview = tk.Button(right_col, text="Grayscale\nPreview", bg="gray", fg="white",
                                          command=self.preview_grayscale)
        self.btn_gray_preview.pack(pady=5, fill=tk.X)

        self.btn_red_preview = tk.Button(right_col, text="Red\nPreview", bg="red", fg="white",
                                         command=lambda: self.preview_channel("red"))
        self.btn_red_preview.pack(pady=5, fill=tk.X)

        self.btn_green_preview = tk.Button(right_col, text="Green\nPreview", bg="green", fg="white",
                                           command=lambda: self.preview_channel("green"))
        self.btn_green_preview.pack(pady=5, fill=tk.X)

        self.btn_blue_preview = tk.Button(right_col, text="Blue\nPreview", bg="blue", fg="white",
                                          command=lambda: self.preview_channel("blue"))
        self.btn_blue_preview.pack(pady=5, fill=tk.X)

        

        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.original_canvas = tk.Canvas(canvas_frame, bg="white")
        self.original_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.decompressed_canvas = tk.Canvas(canvas_frame, bg="white")
        self.decompressed_canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)

        

        bottom_frame = ttk.Frame(self.main_frame, padding=5)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        
        self.info_label_left = ttk.Label(bottom_frame, text="Left image info...")
        self.info_label_left.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.info_label_right = ttk.Label(bottom_frame, text="Right image info...")
        self.info_label_right.pack(side=tk.LEFT, expand=True, fill=tk.X)

        
        self.comparison_frame = ttk.LabelFrame(self.main_frame, text="File Comparison", padding=10)
        self.comparison_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        comp_grid = ttk.Frame(self.comparison_frame)
        comp_grid.pack(fill=tk.X)

        
        ttk.Label(comp_grid, text="Original Image", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(comp_grid, text="Compressed LZW", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(comp_grid, text="Compression Info", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=2)

        
        ttk.Label(comp_grid, text="File Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(comp_grid, text="File Size:").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.orig_size_label = ttk.Label(comp_grid, text="--")
        self.orig_size_label.grid(row=1, column=0, sticky=tk.E, padx=5, pady=2)
        
        self.comp_size_label = ttk.Label(comp_grid, text="--")
        self.comp_size_label.grid(row=1, column=1, sticky=tk.E, padx=5, pady=2)
        
        ttk.Label(comp_grid, text="Ratio:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.ratio_label = ttk.Label(comp_grid, text="--")
        self.ratio_label.grid(row=1, column=2, sticky=tk.E, padx=5, pady=2)

        
        ttk.Label(comp_grid, text="Dimensions:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(comp_grid, text="Dimensions:").grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.orig_dim_label = ttk.Label(comp_grid, text="--")
        self.orig_dim_label.grid(row=2, column=0, sticky=tk.E, padx=5, pady=2)
        
        self.comp_dim_label = ttk.Label(comp_grid, text="--")
        self.comp_dim_label.grid(row=2, column=1, sticky=tk.E, padx=5, pady=2)
        
        ttk.Label(comp_grid, text="Method:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.method_label = ttk.Label(comp_grid, text="--")
        self.method_label.grid(row=2, column=2, sticky=tk.E, padx=5, pady=2)

        
        ttk.Label(comp_grid, text="Entropy:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.entropy_label = ttk.Label(comp_grid, text="--")
        self.entropy_label.grid(row=3, column=2, sticky=tk.E, padx=5, pady=2)

        
        ttk.Label(comp_grid, text="Avg Code Length:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)
        self.avg_code_len_label = ttk.Label(comp_grid, text="--")
        self.avg_code_len_label.grid(row=4, column=2, sticky=tk.E, padx=5, pady=2)

        comp_grid.columnconfigure(0, weight=1)
        comp_grid.columnconfigure(1, weight=1)
        comp_grid.columnconfigure(2, weight=1)

    

        #Görüntü seç ve GUI ye aktar
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("BMP files", "*.bmp"), ("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")]
        )
        if not file_path:
            return
        try:
            from PIL import Image
            self.original_image = Image.open(file_path)
            self.original_image_path = file_path
            self.display_image(self.original_image, self.original_canvas)
            self.info_label_left.config(
                text=f"Opened: {os.path.basename(file_path)} | Size: {self.original_image.width}x{self.original_image.height}"
            )
            

            orig_details = get_image_file_details(file_path)
            if orig_details:
                self.orig_size_label.config(text=f"{orig_details['file_size']/1024:.2f} KB")
                self.orig_dim_label.config(text=orig_details['dimensions'])
                

                self.comp_size_label.config(text="--")
                self.comp_dim_label.config(text="--")
                self.ratio_label.config(text="--")
                self.method_label.config(text="--")
                self.entropy_label.config(text="--")
                self.avg_code_len_label.config(text="--")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")
     



        #Görüntü L ve ya RGB seçeneğine göre sıkıştırma

    def compress_image(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Please open an image first.")
            return
        
        

        if self.is_grayscale.get():
            img = self.original_image.convert("L")
        else:
            img = self.original_image.convert("RGB")
        
        #L görüntüyü numpy array'e dönüştürme 2 boyutlu
        arr = np.array(img)
        if len(arr.shape) == 2:
            

            flat_data = arr.flatten().tolist()
       #RGB görüntüyü numpy array'e dönüştürme 3 boyutlu
        else:
            

            h, w, _ = arr.shape
            flat_data = arr.reshape(h * w * 3).tolist()

        
        use_difference = messagebox.askyesno(
            "Compression Method",
            "Use difference-based compression?\nYes=Difference, No=Standard"
        )
        method = "difference" if use_difference else "standard"


             #RGB görüntü diff metoda dönüştürme
        if method == "difference":
            if self.is_grayscale.get():
                diff_arr = compute_difference_image(arr)
                data = diff_arr.flatten().tolist()
            else:
                r = arr[:, :, 0]
                g = arr[:, :, 1]
                b = arr[:, :, 2]
                r_diff = compute_difference_image(r)
                g_diff = compute_difference_image(g)
                b_diff = compute_difference_image(b)
                hh, ww = r_diff.shape
                data = []
                for i in range(hh):
                    for j in range(ww):
                        data.append(int(r_diff[i, j]))
                        data.append(int(g_diff[i, j]))
                        data.append(int(b_diff[i, j]))
        else:
            data = flat_data

        
        codes = compress_bytes(data)

        
        ent = compute_entropy(data)
        avg_codelen = compute_average_code_length(codes)
        self.last_entropy = ent
        self.last_avg_code_len = avg_codelen

        
        save_path = filedialog.asksaveasfilename(
            title="Save Compressed File",
            defaultextension=".lzw",
            filetypes=[("LZW files", "*.lzw")]
        )
        if not save_path:
            return
        save_compressed_file(save_path, codes, img.size, self.is_grayscale.get(), method)
        self.lzw_file_path = save_path

        
        decompressed_data = decompress_bytes(codes)



        #L görüntü diff metodda bura çalışcak
        if method == "difference":
            if self.is_grayscale.get():
                diff_arr = np.array(decompressed_data, dtype=np.int16).reshape(img.size[1], img.size[0])
                rec_arr = reconstruct_from_difference(diff_arr)
                img_out = np_to_pil(rec_arr, 'L')
            else:
                flat = np.array(decompressed_data, dtype=np.int16)
                hh, ww = img.size[1], img.size[0]
                r_arr = flat[0::3].reshape(hh, ww)
                g_arr = flat[1::3].reshape(hh, ww)
                b_arr = flat[2::3].reshape(hh, ww)
                r_rec = reconstruct_from_difference(r_arr)
                g_rec = reconstruct_from_difference(g_arr)
                b_rec = reconstruct_from_difference(b_arr)
                r_img = np_to_pil(r_rec, 'L')
                g_img = np_to_pil(g_rec, 'L')
                b_img = np_to_pil(b_rec, 'L')
                from PIL import Image
                img_out = Image.merge("RGB", (r_img, g_img, b_img))
        else:
            
            flat = np.array(decompressed_data, dtype=np.uint8)
            if self.is_grayscale.get():
                img_out = Image.fromarray(flat.reshape(img.size[1], img.size[0]), 'L')
            else:
                hh, ww = img.size[1], img.size[0]
                r_data = flat[0::3].reshape(hh, ww)
                g_data = flat[1::3].reshape(hh, ww)
                b_data = flat[2::3].reshape(hh, ww)
                from PIL import Image
                r_img = Image.fromarray(r_data, 'L')
                g_img = Image.fromarray(g_data, 'L')
                b_img = Image.fromarray(b_data, 'L')
                img_out = Image.merge("RGB", (r_img, g_img, b_img))

        self.decompressed_image = img_out
        self.display_image(img_out, self.decompressed_canvas)
        self.info_label_right.config(
            text=f"Compressed preview: {os.path.basename(save_path)}"
        )

        messagebox.showinfo("Success", f"Image compressed and saved to {os.path.basename(save_path)}.")

        
        self.compare_files()



         #.lzw dosyasını yükleme ve decompress
    def load_compressed(self):
        """Load a .lzw file, decompress, and show on the right."""
        file_path = filedialog.askopenfilename( #dosya seç ekranı
            title="Select Compressed File",
            filetypes=[("LZW files", "*.lzw")]
        )


        if not file_path:
            return      #dosya seçmezsem dur
        try:
            self.lzw_file_path = file_path
            codes, img_size, is_gray, method_flag = load_compressed_file(file_path)
            data = decompress_bytes(codes)

            if method_flag == 1:  #diff ise
                if is_gray: #diff gri
                    diff_arr = np.array(data, dtype=np.int16).reshape(img_size[1], img_size[0])
                    rec_arr = reconstruct_from_difference(diff_arr)
                    img_out = np_to_pil(rec_arr, 'L')
                else:  #diff rgb
                    flat = np.array(data, dtype=np.int16)
                    h = img_size[1]
                    w = img_size[0]
                    r_arr = flat[0::3].reshape(h, w)
                    g_arr = flat[1::3].reshape(h, w)
                    b_arr = flat[2::3].reshape(h, w)
                    r_rec = reconstruct_from_difference(r_arr)
                    g_rec = reconstruct_from_difference(g_arr)
                    b_rec = reconstruct_from_difference(b_arr)
                    r_img = np_to_pil(r_rec, 'L')
                    g_img = np_to_pil(g_rec, 'L')
                    b_img = np_to_pil(b_rec, 'L')
                    from PIL import Image
                    img_out = Image.merge("RGB", (r_img, g_img, b_img))
            else:  #diff değil standart
                
                flat = np.array(data, dtype=np.uint8)
                if is_gray:  #standart gri
                    from PIL import Image
                    img_out = Image.fromarray(flat.reshape(img_size[1], img_size[0]), 'L')
                else: #standart rgb
                    h = img_size[1]
                    w = img_size[0]
                    r_data = flat[0::3].reshape(h, w)
                    g_data = flat[1::3].reshape(h, w)
                    b_data = flat[2::3].reshape(h, w)
                    from PIL import Image
                    r_img = Image.fromarray(r_data, 'L')
                    g_img = Image.fromarray(g_data, 'L')
                    b_img = Image.fromarray(b_data, 'L')
                    img_out = Image.merge("RGB", (r_img, g_img, b_img))

            self.decompressed_image = img_out
            self.display_image(img_out, self.decompressed_canvas)
            self.info_label_right.config(
                text=f"Loaded .lzw: {os.path.basename(file_path)} | Size: {img_size[0]}x{img_size[1]}"
            )

            
            self.last_entropy = compute_entropy(data)
            self.last_avg_code_len = compute_average_code_length(codes)
            self.compare_files()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load compressed: {e}")

    def decompress_to_bmp(self):
        """Save the decompressed image as BMP."""
        if not self.decompressed_image:
            messagebox.showwarning("Warning", "No decompressed image to save. Please compress or load a .lzw first.")
            return
       
            #dosya seç ekranı .lzw seç
        save_path = filedialog.asksaveasfilename(
            title="Save Decompressed as BMP",
            defaultextension=".bmp",
            filetypes=[("BMP files", "*.bmp")]
        )
        if not save_path:
            return
        try:
            self.decompressed_image.save(save_path, "BMP")
            messagebox.showinfo("Success", f"Decompressed image saved as {os.path.basename(save_path)}.")
            self.original_image_path = save_path
            self.original_image = self.decompressed_image
            self.compare_files()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save BMP: {str(e)}")




    def preview_grayscale(self):  #gri önizleme
        if not self.original_image:
            messagebox.showwarning("Warning", "No original image loaded.")
            return
        gray_img = self.original_image.convert("L")
        self.display_image(gray_img, self.original_canvas)
        self.info_label_left.config(text=f"Grayscale Preview: {gray_img.width}x{gray_img.height}")


         #RGB renk kanallarına ayır
    def preview_channel(self, channel):
        if not self.original_image:
            messagebox.showwarning("Warning", "No original image loaded.")
            return
        from PIL import Image
        img_rgb = self.original_image.convert("RGB")
        arr = np.array(img_rgb)
        if channel == "red":    #kırmızı önizleme
            arr[:, :, 1] = 0
            arr[:, :, 2] = 0
        elif channel == "green":   #yeşil önizleme
            arr[:, :, 0] = 0
            arr[:, :, 2] = 0
        elif channel == "blue":     #mavi önizleme
            arr[:, :, 0] = 0
            arr[:, :, 1] = 0

        preview_img = Image.fromarray(arr, 'RGB')
        self.display_image(preview_img, self.original_canvas)
        self.info_label_left.config(text=f"{channel.capitalize()} Channel Preview")

    def display_image(self, pil_image, canvas):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 400
            canvas_height = 300
        
        img_ratio = pil_image.width / pil_image.height
        c_ratio = canvas_width / canvas_height
        if img_ratio > c_ratio:
            new_w = canvas_width
            new_h = int(canvas_width / img_ratio)
        else:
            new_h = canvas_height
            new_w = int(canvas_height * img_ratio)

        from PIL import Image, ImageTk
        pil_resized = pil_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_resized)
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_image, anchor=tk.CENTER)

    def compare_files(self):
        """Update the comparison info (original vs compressed)."""
        if self.original_image_path and os.path.exists(self.original_image_path):
            orig_details = get_image_file_details(self.original_image_path)
            if orig_details:
                self.orig_size_label.config(text=f"{orig_details['file_size']/1024:.2f} KB")
                self.orig_dim_label.config(text=orig_details['dimensions'])
        if self.lzw_file_path and os.path.exists(self.lzw_file_path):
            lzw_details = get_lzw_file_details(self.lzw_file_path)
            if lzw_details:
                self.comp_size_label.config(text=f"{lzw_details['file_size']/1024:.2f} KB")
                self.comp_dim_label.config(text=lzw_details['dimensions'])
                self.method_label.config(text=lzw_details['compression_method'])
                if lzw_details['compression_ratio'] > 0:
                    self.ratio_label.config(text=f"{lzw_details['compression_ratio']:.3f}")
                else:
                    self.ratio_label.config(text="--")
                
                if self.last_entropy is not None:
                    self.entropy_label.config(text=f"{self.last_entropy:.3f}")
                else:
                    self.entropy_label.config(text="--")
                if self.last_avg_code_len is not None:
                    self.avg_code_len_label.config(text=f"{self.last_avg_code_len:.3f}")
                else:
                    self.avg_code_len_label.config(text="--")


if __name__ == "__main__":
    root = tk.Tk()
    app = LZWImageCompressorApp(root)
    root.mainloop()
