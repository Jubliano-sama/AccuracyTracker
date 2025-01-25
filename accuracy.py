import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from math import sqrt
from statistics import mean, stdev
from scipy.stats import norm, chi2, binom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ShotAccuracyApp:
    def __init__(self, master):
        self.master = master
        master.title("Shot Accuracy Calculator")
        master.geometry("1600x800")
        
        # Initialize variables
        self.shots = []
        self.avg_coords = None
        self.distances = []
        self.accuracy = None
        self.stdX_dev = None
        self.stdY_dev = None
        self.X_mean = None
        self.Y_mean = None
        self.stdX_error = None
        self.stdY_error = None
        self.total_shots = 0
        self.valid_radius = 15.0  # Initialize valid radius

        # Tkinter Variables for live updates
        self.trials_var = tk.StringVar()
        self.hits_var = tk.StringVar()
        self.radius_var = tk.StringVar(value=str(self.valid_radius))

        # Create frames for better layout
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=10, padx=10, fill=tk.X)

        self.controls_frame = tk.Frame(master)
        self.controls_frame.pack(pady=10, padx=10, fill=tk.X)

        self.metrics_frame = tk.Frame(master)
        self.metrics_frame.pack(pady=10, padx=10, fill=tk.X)

        self.prob_frame = tk.Frame(master)
        self.prob_frame.pack(pady=10, padx=10, fill=tk.X)

        self.plots_frame = tk.Frame(master)
        self.plots_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Shot Entries Treeview
        self.create_shot_entries()

        # Add and Remove Buttons
        self.add_button = tk.Button(self.controls_frame, text="Add Shot", command=self.add_shot)
        self.add_button.grid(row=0, column=0, padx=5)

        self.remove_button = tk.Button(self.controls_frame, text="Remove Selected Shot", command=self.remove_shot)
        self.remove_button.grid(row=0, column=1, padx=5)

        # Import from Excel Button
        self.import_button = tk.Button(self.controls_frame, text="Import from Excel", command=self.import_from_excel)
        self.import_button.grid(row=0, column=2, padx=5)

        # Export to Excel Button
        self.export_button = tk.Button(self.controls_frame, text="Export to Excel", command=self.export_to_excel)
        self.export_button.grid(row=0, column=3, padx=5)

        # Radius input
        self.radius_label = tk.Label(self.controls_frame, text="Target Radius (cm):")
        self.radius_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.radius_entry = tk.Entry(self.controls_frame, textvariable=self.radius_var, width=10)
        self.radius_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Metrics Labels
        self.avg_label = tk.Label(self.metrics_frame, text="Average Coordinates: N/A")
        self.avg_label.grid(row=0, column=0, sticky='w', padx=10)

        self.accuracy_label = tk.Label(self.metrics_frame, text=f"Accuracy (within {self.valid_radius}cm): N/A")
        self.accuracy_label.grid(row=1, column=0, sticky='w', padx=10)

        self.stdX_label = tk.Label(self.metrics_frame, text="Standard Deviation X: N/A")
        self.stdX_label.grid(row=3, column=0, sticky='w', padx=10)

        self.stdY_label = tk.Label(self.metrics_frame, text="Standard Deviation Y: N/A")
        self.stdY_label.grid(row=4, column=0, sticky='w', padx=10)

        # Probability Inputs
        self.trials_label = tk.Label(self.prob_frame, text="Number of Trials:")
        self.trials_label.grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.trials_entry = tk.Entry(self.prob_frame, textvariable=self.trials_var)
        self.trials_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.hits_label = tk.Label(self.prob_frame, text="Number of Hits Within Radius:")
        self.hits_label.grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.hits_entry = tk.Entry(self.prob_frame, textvariable=self.hits_var)
        self.hits_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Probability Labels
        self.prob_xy_label = tk.Label(self.prob_frame, text="Probability of one shot hitting: N/A")
        self.prob_xy_label.grid(row=6, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_binomial_label = tk.Label(self.prob_frame, text="Probability of reaching desired result: N/A")
        self.prob_binomial_label.grid(row=7, column=0, columnspan=2, sticky='w', padx=10)

        # Error Probability Labels
        self.prob_lower_label = tk.Label(self.prob_frame, text="Lower Probability (95% confidence): N/A")
        self.prob_lower_label.grid(row=8, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_higher_label = tk.Label(self.prob_frame, text="Higher Probability (95% confidence): N/A")
        self.prob_higher_label.grid(row=9, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_lower_50_label = tk.Label(self.prob_frame, text="Lower Probability (50% confidence): N/A")
        self.prob_lower_50_label.grid(row=10, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_higher_50_label = tk.Label(self.prob_frame, text="Higher Probability (50% confidence): N/A")
        self.prob_higher_50_label.grid(row=11, column=0, columnspan=2, sticky='w', padx=10)

        # Bind variable changes for live updates
        self.trials_var.trace_add('write', self.on_prob_input_change)
        self.hits_var.trace_add('write', self.on_prob_input_change)
        self.radius_var.trace_add('write', self.on_radius_change)

        # Visualization Plot
        self.setup_visualization_plot()

    def create_shot_entries(self):
        columns = ('#1', '#2')
        self.tree = ttk.Treeview(self.input_frame, columns=columns, show='headings', height=10)
        self.tree.heading('#1', text='X (cm)')
        self.tree.heading('#2', text='Y (cm)')
        self.tree.column('#1', width=100, anchor='center')
        self.tree.column('#2', width=100, anchor='center')
        self.tree.pack(side='left', fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.input_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        self.tree.bind('<ButtonRelease-1>', self.on_shot_change)
        self.tree.bind('<KeyRelease>', self.on_shot_change)

    def add_shot(self):
        add_window = tk.Toplevel(self.master)
        add_window.title("Add New Shot")
        add_window.geometry("300x170")
        add_window.grab_set()

        tk.Label(add_window, text="X Coordinate (cm):").pack(pady=10)
        x_entry = tk.Entry(add_window)
        x_entry.pack()

        tk.Label(add_window, text="Y Coordinate (cm):").pack(pady=10)
        y_entry = tk.Entry(add_window)
        y_entry.pack()

        def submit():
            x = x_entry.get().strip()
            y = y_entry.get().strip()
            if x == "" or y == "":
                messagebox.showerror("Input Error", "Both X and Y coordinates must be filled.")
                return
            try:
                x_val = float(x)
                y_val = float(y)
                self.tree.insert('', 'end', values=(x_val, y_val))
                add_window.destroy()
                self.update_metrics_and_visualization()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numeric values for coordinates.")

        tk.Button(add_window, text="Add Shot", command=submit).pack(pady=10)

    def remove_shot(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Error", "Please select a shot to remove.")
            return
        for item in selected_item:
            self.tree.delete(item)
        self.update_metrics_and_visualization()

    def import_from_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            df_shots = pd.read_excel(file_path)
            if 'X (cm)' not in df_shots.columns or 'Y (cm)' not in df_shots.columns:
                messagebox.showerror("Import Error", "The Excel file must contain 'X (cm)' and 'Y (cm)' columns.")
                return

            for item in self.tree.get_children():
                self.tree.delete(item)

            for _, row in df_shots.iterrows():
                x = row['X (cm)']
                y = row['Y (cm)']
                self.tree.insert('', 'end', values=(x, y))

            self.update_metrics_and_visualization()
            messagebox.showinfo("Import Successful", f"Data imported successfully from {file_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"An error occurred while importing: {e}")

    def on_shot_change(self, event):
        self.update_metrics_and_visualization()

    def update_metrics_and_visualization(self):
        self.calculate_metrics()
        self.calculate_probabilities()
        self.update_visualization()

    def calculate_metrics(self):
        shots = []
        for item in self.tree.get_children():
            x, y = self.tree.item(item)['values']
            shots.append((float(x), float(y)))

        if not shots:
            self.avg_label.config(text="Average Coordinates: N/A")
            self.accuracy_label.config(text=f"Accuracy (within {self.valid_radius}cm): N/A")
            self.stdX_label.config(text="Standard Deviation X: N/A")
            self.stdY_label.config(text="Standard Deviation Y: N/A")
            self.avg_coords = None
            self.distances = []
            self.accuracy = None
            self.std_dev = None
            self.total_shots = 0
            return

        avg_x = mean([shot[0] for shot in shots])
        avg_y = mean([shot[1] for shot in shots])
        self.avg_label.config(text=f"Average Coordinates: ({avg_x:.2f}, {avg_y:.2f}) cm")

        distances = [sqrt((shot[0] - avg_x) ** 2 + (shot[1] - avg_y) ** 2) for shot in shots]
        self.distances = distances

        within_radius = sum(1 for d in distances if d <= self.valid_radius)
        self.accuracy = within_radius
        self.total_shots = len(shots)
        self.accuracy_label.config(text=f"Accuracy (within {self.valid_radius}cm): {within_radius} / {self.total_shots}")

        if len(distances) > 1:
            self.stdX_dev = stdev(item[0] for item in shots)
            self.X_mean = mean(item[0] for item in shots)
            self.Y_mean = mean(item[1] for item in shots)
            self.stdY_dev = stdev(item[1] for item in shots)
            self.stdX_label.config(text=f"Standard Deviation X: {self.stdX_dev:.2f} cm")
            self.stdY_label.config(text=f"Standard Deviation Y: {self.stdY_dev:.2f} cm")
            
            self.stdX_error = self.stdX_dev / sqrt(2 * (len(distances) - 1))
            self.stdY_error = self.stdY_dev / sqrt(2 * (len(distances) - 1))
        else:
            self.std_dev = None
            self.stdX_label.config(text="Standard Deviation X: N/A")
            self.stdY_label.config(text="Standard Deviation Y: N/A")

        self.avg_coords = (avg_x, avg_y)

    def calculate_probabilities(self):
        """
        Updates labels for:
        - Probability of one shot hitting (monte carlo estimate)
        - Probability of reaching desired result (binomial)
        - 95% CI bounds for binomial probability
        - 50% CI bounds for binomial probability
        using a parametric bootstrap approach.
        """
        # If we have no shots or no average coords, there's nothing to compute
        if not self.distances or not self.avg_coords:
            self.prob_xy_label.config(text="Probability of one shot hitting: N/A")
            self.prob_binomial_label.config(text="Probability of reaching desired result: N/A")
            self.prob_lower_label.config(text="Lower Probability (95% confidence): N/A")
            self.prob_higher_label.config(text="Higher Probability (95% confidence): N/A")
            self.prob_lower_50_label.config(text="Lower Probability (50% confidence): N/A")
            self.prob_higher_50_label.config(text="Higher Probability (50% confidence): N/A")
            return

        # Parse the user inputs for binomial trials and hits
        trials_str = self.trials_var.get().strip()
        hits_str = self.hits_var.get().strip()

        if trials_str == "" or hits_str == "":
            # Missing input => no calculation
            self.prob_xy_label.config(text="Probability of one shot hitting: N/A")
            self.prob_binomial_label.config(text="Probability of reaching desired result: N/A")
            self.prob_lower_label.config(text="Lower Probability (95% confidence): N/A")
            self.prob_higher_label.config(text="Higher Probability (95% confidence): N/A")
            self.prob_lower_50_label.config(text="Lower Probability (50% confidence): N/A")
            self.prob_higher_50_label.config(text="Higher Probability (50% confidence): N/A")
            return

        # Validate that trials and hits are integers in correct ranges
        try:
            trials = int(trials_str)
            hits = int(hits_str)
            if trials <= 0 or hits < 0 or hits > trials:
                raise ValueError
        except ValueError:
            self.prob_xy_label.config(text="Probability of one shot hitting: Invalid Input")
            self.prob_binomial_label.config(text="Probability of reaching desired result: Invalid Input")
            self.prob_lower_label.config(text="Lower Probability (95% confidence): Invalid Input")
            self.prob_higher_label.config(text="Higher Probability (95% confidence): Invalid Input")
            self.prob_lower_50_label.config(text="Lower Probability (50% confidence): Invalid Input")
            self.prob_higher_50_label.config(text="Higher Probability (50% confidence): Invalid Input")
            return

        # If std dev wasn't computed (e.g., only 1 shot in the dataset), we can't proceed
        if self.stdX_dev is None or self.stdY_dev is None:
            self.prob_xy_label.config(text="Probability of one shot hitting: N/A")
            self.prob_binomial_label.config(text="Probability of reaching desired result: N/A")
            self.prob_lower_label.config(text="Lower Probability (95% confidence): N/A")
            self.prob_higher_label.config(text="Higher Probability (95% confidence): N/A")
            self.prob_lower_50_label.config(text="Lower Probability (50% confidence): N/A")
            self.prob_higher_50_label.config(text="Higher Probability (50% confidence): N/A")
            return

        # Gather the shot data in x_arr, y_arr for bootstrapping
        shots = []
        for item in self.tree.get_children():
            x_val, y_val = self.tree.item(item)['values']
            shots.append((float(x_val), float(y_val)))
        arr = np.array(shots)
        x_arr = arr[:, 0]
        y_arr = arr[:, 1]
        n_data = len(x_arr)

        # Radius for a "hit"
        radius = self.valid_radius

        # ------------------------------------------------------------------
        # 1) Single Monte Carlo estimate of probability for one shot hitting
        # ------------------------------------------------------------------
        n_mc = 10_000  # number of draws in MC
        sim_x = np.random.normal(self.X_mean, self.stdX_dev, size=n_mc)
        sim_y = np.random.normal(self.Y_mean, self.stdY_dev, size=n_mc)
        dists = np.sqrt(sim_x**2 + sim_y**2)

        # Probability that a single shot is within 'radius'
        prob_hit_one_shot = np.mean(dists <= radius)

        # Binomial probability of getting at least `hits` successes in `trials`
        prob_binom_center = 1 - binom.cdf(hits - 1, trials, prob_hit_one_shot)

        # ------------------------------------------------------------------
        # 2) Parametric Bootstrap to form distribution of binomial probabilities
        # ------------------------------------------------------------------
        n_boot = 1000
        boot_binom_probs = []

        for _ in range(n_boot):
            # (a) Sample sigma_x_star from scaled chi-square
            chi2_x = chi2.rvs(df=n_data - 1)
            sigma_x_star_sq = (n_data - 1)*(self.stdX_dev**2)/chi2_x
            sigma_x_star = np.sqrt(sigma_x_star_sq)

            # (b) Sample mu_x_star from Normal( X_mean, sigma_x_star/sqrt(n_data) )
            mu_x_star = np.random.normal(loc=self.X_mean,
                                        scale=sigma_x_star / np.sqrt(n_data))

            # (c) Sample sigma_y_star
            chi2_y = chi2.rvs(df=n_data - 1)
            sigma_y_star_sq = (n_data - 1)*(self.stdY_dev**2)/chi2_y
            sigma_y_star = np.sqrt(sigma_y_star_sq)

            # (d) Sample mu_y_star
            mu_y_star = np.random.normal(loc=self.Y_mean,
                                        scale=sigma_y_star / np.sqrt(n_data))

            # (e) Monte Carlo to estimate prob of hitting
            sim_x_star = np.random.normal(mu_x_star, sigma_x_star, size=n_mc)
            sim_y_star = np.random.normal(mu_y_star, sigma_y_star, size=n_mc)
            dist_star = np.sqrt(sim_x_star**2 + sim_y_star**2)
            p_star = np.mean(dist_star <= radius)

            # (f) Binomial probability for at least 'hits' in 'trials'
            prob_binom_star = 1 - binom.cdf(hits - 1, trials, p_star)
            boot_binom_probs.append(prob_binom_star)

        boot_binom_probs = np.array(boot_binom_probs)

        # Extract 95% CI => 2.5% and 97.5% percentiles
        prob_binom_lower_95 = np.percentile(boot_binom_probs, 2.5)
        prob_binom_higher_95 = np.percentile(boot_binom_probs, 97.5)

        # Extract 50% CI => 25% and 75% percentiles
        prob_binom_lower_50 = np.percentile(boot_binom_probs, 25)
        prob_binom_higher_50 = np.percentile(boot_binom_probs, 75)

        # ------------------------------------------------------------------
        # 3) Update the GUI labels with our new results
        # ------------------------------------------------------------------
        self.prob_xy_label.config(
            text=f"Probability of one shot hitting: {prob_hit_one_shot * 100:.2f}%"
        )
        self.prob_binomial_label.config(
            text=f"Probability of reaching desired result: {prob_binom_center * 100:.2f}%"
        )

        # 95% CI (from bootstrap)
        self.prob_lower_label.config(
            text=f"Lower Probability (95% confidence): {prob_binom_lower_95 * 100:.2f}%"
        )
        self.prob_higher_label.config(
            text=f"Higher Probability (95% confidence): {prob_binom_higher_95 * 100:.2f}%"
        )

        # 50% CI (from bootstrap)
        self.prob_lower_50_label.config(
            text=f"Lower Probability (50% confidence): {prob_binom_lower_50 * 100:.2f}%"
        )
        self.prob_higher_50_label.config(
            text=f"Higher Probability (50% confidence): {prob_binom_higher_50 * 100:.2f}%"
        )


    def export_to_excel(self):
        if not self.shots_data_available():
            messagebox.showerror("No Data", "There is no data to export. Please add shots and calculate metrics first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not file_path:
            return

        data = [(float(self.tree.item(item)['values'][0]), float(self.tree.item(item)['values'][1])) for item in self.tree.get_children()]
        df_shots = pd.DataFrame(data, columns=['X (cm)', 'Y (cm)'])

        try:
            with pd.ExcelWriter(file_path) as writer:
                df_shots.to_excel(writer, sheet_name='Shots', index=False)
            messagebox.showinfo("Export Successful", f"Data exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting: {e}")

    def shots_data_available(self):
        return self.tree.get_children() and self.avg_coords is not None

    def setup_visualization_plot(self):
        self.visualization_frame = tk.Frame(self.plots_frame)
        self.visualization_frame.pack(side='left', fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.density_frame = tk.Frame(self.plots_frame)
        self.density_frame.pack(side='right', fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Visualization Grid Plot
        self.fig_vis, self.ax_vis = plt.subplots(figsize=(6,6))
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.visualization_frame)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_visualization()

        # Separate KDE plots for X and Y
        self.fig_density, (self.ax_density_x, self.ax_density_y) = plt.subplots(2, 1, figsize=(6,6))
        self.canvas_density = FigureCanvasTkAgg(self.fig_density, master=self.density_frame)
        self.canvas_density.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_density_plot()

    def setup_visualization(self):
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)
        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

    def setup_density_plot(self):
        self.ax_density_x.set_title('X Coordinate Density')
        self.ax_density_x.set_xlabel('X (cm)')
        self.ax_density_x.set_ylabel('Density')
        self.ax_density_x.grid(True)

        self.ax_density_y.set_title('Y Coordinate Density')
        self.ax_density_y.set_xlabel('Y (cm)')
        self.ax_density_y.set_ylabel('Density')
        self.ax_density_y.grid(True)

        self.fig_density.tight_layout()
        self.canvas_density.draw()

    def update_visualization(self):
        self.ax_vis.clear()
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)

        shots = []
        for item in self.tree.get_children():
            x, y = self.tree.item(item)['values']
            shots.append((float(x), float(y)))

        if shots:
            x_vals = [shot[0] for shot in shots]
            y_vals = [shot[1] for shot in shots]
            self.ax_vis.scatter(x_vals, y_vals, c='blue', label='Shots')

            if self.avg_coords:
                self.ax_vis.scatter(self.avg_coords[0], self.avg_coords[1], c='green', marker='x', s=100, label='Center')
                circle = Circle(self.avg_coords, self.valid_radius, color='red', fill=False, linestyle='--', label=f'{self.valid_radius}cm Radius')
                self.ax_vis.add_patch(circle)

            self.ax_vis.legend()
            self.ax_vis.set_aspect('equal', adjustable='datalim')

        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        if len(shots) > 2:
            from scipy.stats import gaussian_kde
            x_vals = [shot[0] for shot in shots]
            y_vals = [shot[1] for shot in shots]

            self.ax_density_x.clear()
            kde_x = gaussian_kde(x_vals, bw_method='scott')
            x_range = np.linspace(min(x_vals) - 10, max(x_vals) + 10, 1000)
            y_kde_x = kde_x(x_range)
            self.ax_density_x.plot(x_range, y_kde_x, color='blue', lw=2)
            self.ax_density_x.fill_between(x_range, y_kde_x, color='skyblue', alpha=0.5)
            self.ax_density_x.set_title('X Coordinate Density')
            self.ax_density_x.set_xlabel('X (cm)')
            self.ax_density_x.set_ylabel('Density')
            self.ax_density_x.grid(True)

            self.ax_density_y.clear()
            kde_y = gaussian_kde(y_vals, bw_method='scott')
            y_range = np.linspace(min(y_vals) - 10, max(y_vals) + 10, 1000)
            y_kde_y = kde_y(y_range)
            self.ax_density_y.plot(y_range, y_kde_y, color='blue', lw=2)
            self.ax_density_y.fill_between(y_range, y_kde_y, color='skyblue', alpha=0.5)
            self.ax_density_y.set_title('Y Coordinate Density')
            self.ax_density_y.set_xlabel('Y (cm)')
            self.ax_density_y.set_ylabel('Density')
            self.ax_density_y.grid(True)

        self.fig_density.tight_layout()
        self.canvas_density.draw()

    def on_radius_change(self, *args):
        try:
            new_radius = float(self.radius_var.get())
            if new_radius <= 0:
                raise ValueError("Radius must be positive")
            self.valid_radius = new_radius
            self.update_metrics_and_visualization()
        except ValueError:
            self.radius_var.set(str(self.valid_radius))

    def on_prob_input_change(self, *args):
        self.calculate_probabilities()

def main():
    root = tk.Tk()
    root.state('zoomed')
    app = ShotAccuracyApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()