# Vue graphique pour la vérification des changements de vue.

import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image

def user_verification_interface(images, changes):
    """
    Vérification manuelle des changements de vue détectés avec interface graphique.
    """
    def on_confirm():
        highlight_index(current_idx.get(), "green")
        confirmed_changes.append(changes[current_idx.get()])
        next_image()
    
    def on_skip():
        highlight_index(current_idx.get(), "red")
        next_image()

    def next_image():
        current_idx.set(current_idx.get() + 1)
        idx = current_idx.get()
        if idx < len(changes):
            display_images(changes[idx])
        else:
            messagebox.showinfo("Fin de la vérification", "Tous les changements de vue ont été vérifiés.")
            root.quit()

    def highlight_index(index, color):
        """ Surligne l'indice dans la couleur spécifiée """
        start_pos = changes_text.search(str(changes[index]), "1.0", stopindex="end")
        end_pos = f"{start_pos}+{len(str(changes[index]))}c"
        changes_text.tag_add(str(index), start_pos, end_pos)
        changes_text.tag_config(str(index), background=color)

    confirmed_changes = []

    root = ctk.CTk()
    root.title("Vérification des changements de vue")
    root.minsize(800, 600)

    # Partie gauche : informations et boutons
    # Partie droite : images
    left_frame = ctk.CTkFrame(root)
    right_frame = ctk.CTkFrame(root)

    # Placement des frames
    left_frame.place(relx=0.0, rely=0.0, relwidth=0.2, relheight=1.0)
    right_frame.place(relx=0.2, rely=0.0, relwidth=0.8, relheight=1.0)

    # Partie gauche
    left_frame.columnconfigure((0, 1), weight=1)
    left_frame.rowconfigure((0, 1, 2), weight=1)
    
    # Création des labels et boutons
    changes_text = tk.Text(left_frame, wrap="word", state="disabled", width=100, height=5)   
    left_question_label = ctk.CTkLabel(left_frame, text="Changement de vue ?", anchor="s")
    left_boutton_oui = ctk.CTkButton(left_frame, text="Oui", command=on_confirm)
    left_boutton_non = ctk.CTkButton(left_frame, text="Non", command=on_skip)

    changes_text.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    left_question_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    left_boutton_oui.grid(row=2, column=0, sticky="ne")
    left_boutton_non.grid(row=2, column=1, sticky="nw")
    
    changes_text.configure(state="normal")
    changes_text.tag_configure("center", justify="center")
    changes_text.insert(1.0, "Changements de vue détectés :\n")
    changes_text.insert(3.0, " ".join(map(str, changes)))
    changes_text.configure(state="disabled")

    # Partie droite
    right_frame.columnconfigure((0, 1), weight=1)
    right_frame.rowconfigure(0, weight=2)

    img1_container = ctk.CTkFrame(right_frame)
    img2_container = ctk.CTkFrame(right_frame)

    img1_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    img2_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

    img1_label = ctk.CTkLabel(img1_container, text="", image=None)
    img2_label = ctk.CTkLabel(img2_container, text="", image=None)

    img1_label.pack(expand=True)
    img2_label.pack(expand=True)

    def display_images(idx):
        img1 = Image.fromarray(cv2.cvtColor(images[idx-1], cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

        size = (300, 300)
        img1_ctk = ctk.CTkImage(light_image=img1, size=size)
        img2_ctk = ctk.CTkImage(light_image=img2, size=size)

        img1_label.configure(image=img1_ctk)
        img1_label.image = img1_ctk
        img2_label.configure(image=img2_ctk)
        img2_label.image = img2_ctk

    def update_labels():
        # Mise à jour dynamique des labels avec la largeur actuelle du frame
        left_question_label.configure(wraplength=left_frame.winfo_width())
        root.after(100, update_labels)  # Répéter après 100ms

    root.after(100, update_labels)  # Démarrer la mise à jour dynamique après 100ms
    
    current_idx = ctk.IntVar(value=0)

    display_images(changes[0])

    root.mainloop()
    return confirmed_changes