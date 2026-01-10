import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
import pandas as pd


class ElectricityConsumptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prédiction de Consommation Électrique - LIU Data Mining")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Données par défaut
        self.default_data = {
            "temp": [5, 7, 9, 12, 15, 17, 20, 22],
            "consumption": [52, 48, 45, 41, 38, 35, 32, 30]
        }

        self.df = pd.DataFrame(self.default_data)
        self.model = None
        self.canvas = None
        self.current_plot = None

        # Style
        self.setup_styles()

        self.create_widgets()
        self.calculate_regression()
        self.show_regression_plot()  # Afficher directement le graphique

    def setup_styles(self):
        """Configure les styles pour une meilleure apparence"""
        style = ttk.Style()
        style.theme_use('clam')

        # Style pour les boutons
        style.configure('Action.TButton',
                        font=('Arial', 10, 'bold'),
                        padding=10)

        # Style pour le Treeview
        style.configure('Custom.Treeview',
                        font=('Arial', 10),
                        rowheight=30)
        style.configure('Custom.Treeview.Heading',
                        font=('Arial', 11, 'bold'))

    def create_widgets(self):
        # En-tête avec informations de l'exposé
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=15)
        header_frame.pack(fill="x")

        tk.Label(header_frame,
                 text="Prédiction de Consommation Électrique",
                 font=("Arial", 20, "bold"),
                 bg="#2c3e50",
                 fg="white").pack()

        tk.Label(header_frame,
                 text="Exposé par: Abdellahi Ahmed Mreizigue (12430003) | Moctar Ely (12430015) | Ahmed Salem Aref (12430106)",
                 font=("Arial", 10),
                 bg="#2c3e50",
                 fg="#ecf0f1").pack(pady=(5, 0))

        # Frame principal avec 2 colonnes
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # COLONNE GAUCHE - Données et contrôles
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.pack(side="left", fill="both", padx=(0, 5))

        # Cadre pour les données
        data_frame = tk.LabelFrame(left_frame,
                                   text="📊 Données d'Entraînement",
                                   font=("Arial", 12, "bold"),
                                   bg="white",
                                   padx=10,
                                   pady=10)
        data_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Tableau avec scrollbar
        tree_frame = tk.Frame(data_frame, bg="white")
        tree_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")

        columns = ("temp", "consumption")
        self.tree = ttk.Treeview(tree_frame,
                                 columns=columns,
                                 show="headings",
                                 height=10,
                                 style='Custom.Treeview',
                                 yscrollcommand=scrollbar.set)

        self.tree.heading("temp", text="🌡️ Température (°C)")
        self.tree.heading("consumption", text="⚡ Consommation (kWh/jour)")

        self.tree.column("temp", width=150, anchor="center")
        self.tree.column("consumption", width=180, anchor="center")

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.tree.yview)

        self.update_table()

        # Cadre d'ajout de données
        add_frame = tk.Frame(data_frame, bg="white")
        add_frame.pack(pady=10)

        tk.Label(add_frame, text="Température (°C):", bg="white", font=("Arial", 10)).grid(row=0, column=0, padx=5,
                                                                                           sticky="e")
        self.temp_entry = ttk.Entry(add_frame, width=12, font=("Arial", 10))
        self.temp_entry.grid(row=0, column=1, padx=5)

        tk.Label(add_frame, text="Consommation (kWh/j):", bg="white", font=("Arial", 10)).grid(row=0, column=2, padx=5,
                                                                                               sticky="e")
        self.consumption_entry = ttk.Entry(add_frame, width=12, font=("Arial", 10))
        self.consumption_entry.grid(row=0, column=3, padx=5)

        # Boutons d'action
        btn_frame = tk.Frame(data_frame, bg="white")
        btn_frame.pack(pady=10)

        buttons = [
            ("➕ Ajouter", self.add_data, "#27ae60"),
            ("🗑️ Supprimer", self.delete_data, "#e74c3c"),
            ("🔄 Réinitialiser", self.reset_data, "#f39c12"),
            ("📈 Statistiques", self.show_statistics, "#3498db")
        ]

        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(btn_frame,
                            text=text,
                            command=command,
                            bg=color,
                            fg="white",
                            font=("Arial", 10, "bold"),
                            relief="flat",
                            padx=15,
                            pady=8,
                            cursor="hand2")
            btn.grid(row=0, column=i, padx=5)

            # Effet hover
            btn.bind("<Enter>", lambda e, b=btn: b.config(relief="raised"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(relief="flat"))

        # Cadre de prédiction
        pred_frame = tk.LabelFrame(left_frame,
                                   text="🔮 Prédiction",
                                   font=("Arial", 12, "bold"),
                                   bg="white",
                                   padx=15,
                                   pady=15)
        pred_frame.pack(fill="x")

        tk.Label(pred_frame,
                 text="Entrez une température pour prédire la consommation :",
                 bg="white",
                 font=("Arial", 10)).pack(pady=(0, 10))

        input_frame = tk.Frame(pred_frame, bg="white")
        input_frame.pack()

        tk.Label(input_frame, text="Température (°C):", bg="white", font=("Arial", 10, "bold")).pack(side="left",
                                                                                                     padx=5)
        self.pred_temp_entry = ttk.Entry(input_frame, width=15, font=("Arial", 11))
        self.pred_temp_entry.pack(side="left", padx=5)

        pred_btn = tk.Button(input_frame,
                             text="🎯 Prédire",
                             command=self.predict_consumption,
                             bg="#9b59b6",
                             fg="white",
                             font=("Arial", 10, "bold"),
                             relief="flat",
                             padx=20,
                             pady=8,
                             cursor="hand2")
        pred_btn.pack(side="left", padx=5)
        pred_btn.bind("<Enter>", lambda e: pred_btn.config(relief="raised"))
        pred_btn.bind("<Leave>", lambda e: pred_btn.config(relief="flat"))

        self.prediction_result = tk.Label(pred_frame,
                                          text="",
                                          font=("Arial", 13, "bold"),
                                          bg="white",
                                          fg="#2c3e50")
        self.prediction_result.pack(pady=15)

        # Équation de régression
        self.equation_frame = tk.Frame(left_frame, bg="#ecf0f1", relief="solid", borderwidth=1)
        self.equation_frame.pack(fill="x", pady=(10, 0))

        self.equation_label = tk.Label(self.equation_frame,
                                       text="",
                                       font=("Arial", 11, "bold"),
                                       bg="#ecf0f1",
                                       fg="#2c3e50",
                                       pady=10)
        self.equation_label.pack()

        # COLONNE DROITE - Graphiques
        right_frame = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Titre du graphique
        graph_header = tk.Frame(right_frame, bg="#34495e", pady=10)
        graph_header.pack(fill="x")

        tk.Label(graph_header,
                 text="📊 Visualisation",
                 font=("Arial", 14, "bold"),
                 bg="#34495e",
                 fg="white").pack()

        # Boutons de changement de vue
        view_btn_frame = tk.Frame(right_frame, bg="white", pady=10)
        view_btn_frame.pack()

        views = [
            ("📍 Nuage de Points", self.show_scatter_plot, "#16a085"),
            ("📈 Droite de Régression", self.show_regression_plot, "#c0392b"),
            ("📊 Graphique Complet", self.show_combined_plot, "#8e44ad")
        ]

        for text, command, color in views:
            btn = tk.Button(view_btn_frame,
                            text=text,
                            command=command,
                            bg=color,
                            fg="white",
                            font=("Arial", 10, "bold"),
                            relief="flat",
                            padx=15,
                            pady=8,
                            cursor="hand2")
            btn.pack(side="left", padx=5)
            btn.bind("<Enter>", lambda e, b=btn: b.config(relief="raised"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(relief="flat"))

        # Frame pour le graphique intégré
        self.plot_frame = tk.Frame(right_frame, bg="white")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def update_table(self):
        """Met à jour le tableau avec les données actuelles"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for _, row in self.df.iterrows():
            self.tree.insert("", tk.END, values=(f"{row['temp']:.1f}", f"{row['consumption']:.1f}"))

    def add_data(self):
        """Ajoute une nouvelle donnée"""
        try:
            temp = float(self.temp_entry.get())
            consumption = float(self.consumption_entry.get())

            new_data = pd.DataFrame({"temp": [temp], "consumption": [consumption]})
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            self.df = self.df.sort_values('temp').reset_index(drop=True)

            self.update_table()
            self.calculate_regression()
            self.update_current_plot()

            self.temp_entry.delete(0, tk.END)
            self.consumption_entry.delete(0, tk.END)

            messagebox.showinfo("✅ Succès", "Donnée ajoutée avec succès!")

        except ValueError:
            messagebox.showerror("❌ Erreur", "Veuillez entrer des valeurs numériques valides")

    def delete_data(self):
        """Supprime la donnée sélectionnée"""
        selected_item = self.tree.selection()
        if selected_item:
            index = self.tree.index(selected_item[0])
            self.df = self.df.drop(index).reset_index(drop=True)
            self.update_table()
            self.calculate_regression()
            self.update_current_plot()
            messagebox.showinfo("✅ Succès", "Donnée supprimée avec succès!")
        else:
            messagebox.showwarning("⚠️ Attention", "Veuillez sélectionner une ligne à supprimer")

    def reset_data(self):
        """Réinitialise aux données par défaut"""
        self.df = pd.DataFrame(self.default_data)
        self.update_table()
        self.calculate_regression()
        self.update_current_plot()
        messagebox.showinfo("🔄 Réinitialisation", "Données réinitialisées aux valeurs par défaut")

    def calculate_regression(self):
        """Calcule la régression linéaire"""
        if len(self.df) < 2:
            self.equation_label.config(text="⚠️ Pas assez de données pour calculer la régression")
            return

        X = self.df["temp"].values.reshape(-1, 1)
        y = self.df["consumption"].values

        self.model = LinearRegression()
        self.model.fit(X, y)

        coeff = self.model.coef_[0]
        intercept = self.model.intercept_
        r_squared = self.model.score(X, y)

        equation = f"📐 Équation : Y = {coeff:.3f}X + {intercept:.3f}  |  R² = {r_squared:.3f}"
        self.equation_label.config(text=equation)

    def show_statistics(self):
        """Affiche les statistiques détaillées"""
        if len(self.df) == 0:
            messagebox.showinfo("📊 Statistiques", "Aucune donnée disponible")
            return

        correlation = self.df['temp'].corr(self.df['consumption'])

        stats = f"""
╔══════════════════════════════════════════╗
║     STATISTIQUES DESCRIPTIVES            ║
╚══════════════════════════════════════════╝

📊 Nombre d'observations : {len(self.df)}

🌡️  TEMPÉRATURE :
   • Minimum     : {self.df['temp'].min():.2f} °C
   • Maximum     : {self.df['temp'].max():.2f} °C
   • Moyenne     : {self.df['temp'].mean():.2f} °C
   • Écart-type  : {self.df['temp'].std():.2f} °C

⚡ CONSOMMATION :
   • Minimum     : {self.df['consumption'].min():.2f} kWh/jour
   • Maximum     : {self.df['consumption'].max():.2f} kWh/jour
   • Moyenne     : {self.df['consumption'].mean():.2f} kWh/jour
   • Écart-type  : {self.df['consumption'].std():.2f} kWh/jour

🔗 CORRÉLATION : {correlation:.3f}
   {'   (Corrélation négative forte)' if correlation < -0.7 else ''}

💡 INTERPRÉTATION :
   • La corrélation de {correlation:.3f} indique une relation
     {'forte' if abs(correlation) > 0.7 else 'modérée'} et {'négative' if correlation < 0 else 'positive'}.
   • Quand la température {'augmente' if correlation < 0 else 'diminue'}, 
     la consommation {'diminue' if correlation < 0 else 'augmente'}.
        """

        messagebox.showinfo("📈 Statistiques Descriptives", stats)

    def predict_consumption(self):
        """Prédit la consommation pour une température donnée"""
        if self.model is None:
            messagebox.showerror("❌ Erreur", "Veuillez d'abord calculer la régression")
            return

        try:
            temp = float(self.pred_temp_entry.get())
            prediction = self.model.predict([[temp]])[0]

            result_text = f"🎯 Pour {temp}°C :\nConsommation prédite = {prediction:.2f} kWh/jour"
            self.prediction_result.config(text=result_text, fg="#27ae60")

        except ValueError:
            messagebox.showerror("❌ Erreur", "Veuillez entrer une température valide")

    def clear_plot(self):
        """Efface le graphique actuel"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def show_scatter_plot(self):
        """Affiche le nuage de points intégré"""
        self.clear_plot()
        self.current_plot = 'scatter'

        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
        ax.scatter(self.df["temp"], self.df["consumption"], color='#3498db', s=100, alpha=0.7, edgecolors='black')
        ax.set_title("Nuage de Points: Température vs Consommation", fontsize=14, fontweight='bold')
        ax.set_xlabel("Température (°C)", fontsize=12)
        ax.set_ylabel("Consommation (kWh/jour)", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_regression_plot(self):
        """Affiche la droite de régression intégrée"""
        if self.model is None:
            messagebox.showerror("❌ Erreur", "Veuillez d'abord calculer la régression")
            return

        self.clear_plot()
        self.current_plot = 'regression'

        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

        ax.scatter(self.df["temp"], self.df["consumption"], color='#3498db', s=100, alpha=0.7, edgecolors='black',
                   label="Données", zorder=3)

        X_vals = np.linspace(self.df["temp"].min() - 2, self.df["temp"].max() + 2, 100).reshape(-1, 1)
        y_vals = self.model.predict(X_vals)
        ax.plot(X_vals, y_vals, color='#e74c3c', linewidth=3, label="Droite de régression", zorder=2)

        ax.set_title("Régression Linéaire: Température vs Consommation", fontsize=14, fontweight='bold')
        ax.set_xlabel("Température (°C)", fontsize=12)
        ax.set_ylabel("Consommation (kWh/jour)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_combined_plot(self):
        """Affiche un graphique combiné avec résidus"""
        if self.model is None:
            messagebox.showerror("❌ Erreur", "Veuillez d'abord calculer la régression")
            return

        self.clear_plot()
        self.current_plot = 'combined'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

        # Graphique principal
        ax1.scatter(self.df["temp"], self.df["consumption"], color='#3498db', s=100, alpha=0.7, edgecolors='black',
                    label="Données observées", zorder=3)

        X_vals = np.linspace(self.df["temp"].min() - 2, self.df["temp"].max() + 2, 100).reshape(-1, 1)
        y_vals = self.model.predict(X_vals)
        ax1.plot(X_vals, y_vals, color='#e74c3c', linewidth=3, label="Droite de régression", zorder=2)

        # Lignes de résidus
        X = self.df["temp"].values.reshape(-1, 1)
        y_pred = self.model.predict(X)
        for i in range(len(self.df)):
            ax1.plot([self.df["temp"].iloc[i], self.df["temp"].iloc[i]],
                     [self.df["consumption"].iloc[i], y_pred[i]],
                     'g--', alpha=0.5, linewidth=1, zorder=1)

        ax1.set_title("Régression avec Résidus", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Température (°C)", fontsize=11)
        ax1.set_ylabel("Consommation (kWh/jour)", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Graphique des résidus
        residuals = self.df["consumption"].values - y_pred
        ax2.scatter(self.df["temp"], residuals, color='#27ae60', s=100, alpha=0.7, edgecolors='black')
        ax2.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
        ax2.set_title("Analyse des Résidus", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Température (°C)", fontsize=11)
        ax2.set_ylabel("Résidus", fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_current_plot(self):
        """Met à jour le graphique actuel après modification des données"""
        if self.current_plot == 'scatter':
            self.show_scatter_plot()
        elif self.current_plot == 'regression':
            self.show_regression_plot()
        elif self.current_plot == 'combined':
            self.show_combined_plot()


def main():
    root = tk.Tk()
    app = ElectricityConsumptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
