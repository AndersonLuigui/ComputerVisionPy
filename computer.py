from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import cv2
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# CARREGAR O MODELO MOBILENETV2 PRE TREINADO
print("CARREGANDO MODELO DE IA MobileNetV2...")
modelo = MobileNetV2(weights='imagenet')

def processar_imagem(imagem_path):
    """Processa a imagem selecionada"""
    try:
        # CARREGAR A IMAGEM USANDO O OPEN CV
        imagem = cv2.imread(imagem_path)
        if imagem is None:
            messagebox.showerror("Erro", f"Erro ao carregar a imagem {imagem_path}")
            return
        
        print(f"Processando imagem: {imagem_path}")
        
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  # CONVERTE PARA RGB
        
        # PREPROCESSAR A IMAGEM PARA O FORMATO ESPERADO PELO MODELO
        imagem_redimensionada = cv2.resize(imagem_rgb, (224, 224))  # REDIMENSIONA PARA 224x224
        imagem_array = np.expand_dims(imagem_redimensionada, axis=0)  # ADICIONA UMA DIMENSÃO PARA O BATCH
        imagem_array = preprocess_input(imagem_array)  # PREPROCESSA A IMAGEM
        
        # FAZER A PREVISÃO
        print("Executando previsão...")
        predicoes = modelo.predict(imagem_array)
        label = decode_predictions(predicoes)
        
        resultado = f"Objeto identificado: {label[0][0][1]}\nConfiança: {label[0][0][2]*100:.2f}%"
        print(resultado)
        messagebox.showinfo("Resultado", resultado)
        
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar: {str(e)}")

def selecionar_imagem():
    """ABRE MODAL PARA CARREGAR A IMAGEM"""
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    
    arquivo = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
    )
    
    root.destroy()
    
    if arquivo:
        processar_imagem(arquivo)

# interface gráfica
root = tk.Tk()
root.title("Computer Vision - Anderson Luigui - MobileNetV2")
root.geometry("500x300")

# Título
titulo = ttk.Label(root, text="Classificador de Imagens com IA", font=("Arial", 16, "bold"))
titulo.pack(pady=20)

# Instrução
instrucao = ttk.Label(root, text="Clique no botão abaixo para selecionar uma imagem", font=("Arial", 10))
instrucao.pack(pady=10)

# Botão para selecionar imagem
botao = ttk.Button(root, text="📁 Carregar Imagem", command=selecionar_imagem)
botao.pack(pady=20, padx=20, fill=tk.X)

# Label para mostrar arquivo selecionado
arquivo_label = ttk.Label(root, text="Nenhuma imagem selecionada", foreground="gray")
arquivo_label.pack(pady=10)

# Botão sair
botao_sair = ttk.Button(root, text="Sair", command=root.quit)
botao_sair.pack(pady=20, padx=20, fill=tk.X)

root.mainloop()