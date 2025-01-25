import argparse
import sys
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

###############################################################################
# 1) ShotsData: a “logic” class that holds shot data & performs all calculations
###############################################################################

class ShotsData:
    def __init__(self, radius=15.0, seed=42):
        np.random.seed(seed=seed)
        self.shots = []  # list of (x, y) floats
        self.valid_radius = radius
        
        # Derived metrics
        self.avg_coords = None  # (mean_x, mean_y)
        self.distances = []     # radial distances from the mean
        self.accuracy = None
        self.stdX_dev = None
        self.stdY_dev = None
        self.X_mean = None
        self.Y_mean = None
        self.stdX_error = None
        self.stdY_error = None
        self.total_shots = 0
        
        # Probability inputs
        self.trials = None
        self.hits = None
        
        # Probability results (95% etc.)
        self.prob_hit_one_shot = None
        self.prob_binomial = None
        self.prob_binomial_lower_95 = None
        self.prob_binomial_higher_95 = None
        self.prob_binomial_lower_50 = None
        self.prob_binomial_higher_50 = None
    
    def add_shot(self, x, y):
        self.shots.append((float(x), float(y)))
    
    def remove_shot(self, index):
        """Remove shot by list index (0-based)."""
        if 0 <= index < len(self.shots):
            del self.shots[index]
    
    def list_shots(self):
        """Return list of all shots as (index, x, y)."""
        return [(i, s[0], s[1]) for i, s in enumerate(self.shots)]
    
    def import_from_excel(self, path):
        """Load shots from an Excel file with columns 'X (cm)' and 'Y (cm)'."""
        df_shots = pd.read_excel(path)
        if 'X (cm)' not in df_shots.columns or 'Y (cm)' not in df_shots.columns:
            raise ValueError("Excel must have columns 'X (cm)' and 'Y (cm)'")
        self.shots.clear()
        for _, row in df_shots.iterrows():
            x = float(row['X (cm)'])
            y = float(row['Y (cm)'])
            self.shots.append((x, y))
    
    def export_to_excel(self, path):
        """Save current shots to an Excel file with columns 'X (cm)', 'Y (cm)'."""
        if not self.shots:
            raise ValueError("No shots data to export.")
        df_shots = pd.DataFrame(self.shots, columns=['X (cm)', 'Y (cm)'])
        with pd.ExcelWriter(path) as writer:
            df_shots.to_excel(writer, sheet_name='Shots', index=False)
    
    def set_radius(self, radius):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.valid_radius = float(radius)
    
    def set_trials(self, trials):
        if trials < 0:
            raise ValueError("Trials must be >= 0.")
        self.trials = trials
    
    def set_hits(self, hits):
        if hits < 0:
            raise ValueError("Hits must be >= 0.")
        self.hits = hits
    
    def calculate_metrics(self):
        """Compute all the basic metrics: average, std dev, accuracy, etc."""
        if not self.shots:
            # Clear out any old metrics
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
            return
        
        x_vals = [s[0] for s in self.shots]
        y_vals = [s[1] for s in self.shots]
        
        avg_x = mean(x_vals)
        avg_y = mean(y_vals)
        self.avg_coords = (avg_x, avg_y)
        
        distances = [sqrt((sx - avg_x)**2 + (sy - avg_y)**2) for sx, sy in self.shots]
        self.distances = distances
        
        within_radius = sum(1 for d in distances if d <= self.valid_radius)
        self.accuracy = within_radius
        self.total_shots = len(self.shots)
        
        # Standard deviation (stdev) only valid if 2+ points
        if len(self.shots) > 1:
            self.stdX_dev = stdev(x_vals)
            self.stdY_dev = stdev(y_vals)
            self.X_mean = avg_x
            self.Y_mean = avg_y
            # For older usage
            n = len(self.shots)
            self.stdX_error = self.stdX_dev / sqrt(2 * (n - 1))
            self.stdY_error = self.stdY_dev / sqrt(2 * (n - 1))
        else:
            self.stdX_dev = None
            self.stdY_dev = None
            self.X_mean = avg_x
            self.Y_mean = avg_y
            self.stdX_error = None
            self.stdY_error = None
    
    def calculate_probabilities(self):
        """
        Compute all probability & confidence interval metrics
        based on self.trials and self.hits (if set).
        """
        self.prob_hit_one_shot = None
        self.prob_binomial = None
        self.prob_binomial_lower_95 = None
        self.prob_binomial_higher_95 = None
        self.prob_binomial_lower_50 = None
        self.prob_binomial_higher_50 = None
        
        # If no shots or only 1 shot => we can't do meaningful stdev-based simulation
        if (not self.shots) or (self.stdX_dev is None or self.stdY_dev is None):
            return
        if self.trials is None or self.hits is None:
            return
        
        # "hit" radius
        radius = self.valid_radius
        
        # Single Monte Carlo estimate
        n_mc = 10_000
        sim_x = np.random.normal(self.X_mean, self.stdX_dev, size=n_mc)
        sim_y = np.random.normal(self.Y_mean, self.stdY_dev, size=n_mc)
        
        dists = np.sqrt((sim_x - self.X_mean)**2 + (sim_y - self.Y_mean)**2)
        prob_hit_one_shot = np.mean(dists <= radius)
        self.prob_hit_one_shot = prob_hit_one_shot
        
        # Probability of >= hits in binomial
        trials = self.trials
        hits = self.hits
        prob_binom_center = 1 - binom.cdf(hits - 1, trials, prob_hit_one_shot)
        self.prob_binomial = prob_binom_center
        
        # Parametric bootstrap for confidence intervals
        shots_arr = np.array(self.shots)
        n_data = len(shots_arr)
        
        n_boot = 10_000
        boot_binom_probs = []
        
        for _ in range(n_boot):
            # X dimension
            chi2_x = chi2.rvs(df=n_data - 1)
            sigma_x_star_sq = (n_data - 1)*(self.stdX_dev**2)/chi2_x
            sigma_x_star = np.sqrt(sigma_x_star_sq)
            mu_x_star = np.random.normal(loc=self.X_mean, scale=sigma_x_star / np.sqrt(n_data))
            
            # Y dimension
            chi2_y = chi2.rvs(df=n_data - 1)
            sigma_y_star_sq = (n_data - 1)*(self.stdY_dev**2)/chi2_y
            sigma_y_star = np.sqrt(sigma_y_star_sq)
            mu_y_star = np.random.normal(loc=self.Y_mean, scale=sigma_y_star / np.sqrt(n_data))
            
            # Sim
            sim_x_star = np.random.normal(mu_x_star, sigma_x_star, size=n_mc)
            sim_y_star = np.random.normal(mu_y_star, sigma_y_star, size=n_mc)
            dist_star = np.sqrt((sim_x_star - mu_x_star)**2 + (sim_y_star - mu_y_star)**2)
            p_star = np.mean(dist_star <= radius)
            
            prob_binom_star = 1 - binom.cdf(hits - 1, trials, p_star)
            boot_binom_probs.append(prob_binom_star)
        
        boot_binom_probs = np.array(boot_binom_probs)
        self.prob_binomial_lower_95 = np.percentile(boot_binom_probs, 2.5)
        self.prob_binomial_higher_95 = np.percentile(boot_binom_probs, 97.5)
        self.prob_binomial_lower_50 = np.percentile(boot_binom_probs, 25)
        self.prob_binomial_higher_50 = np.percentile(boot_binom_probs, 75)

###############################################################################
# 2) The GUI class (mostly your original code, but uses ShotsData internally)
###############################################################################

class ShotAccuracyApp:
    def __init__(self, master):
        self.master = master
        master.title("Shot Accuracy Calculator")
        master.geometry("1600x800")
        
        # Now keep a ShotsData instance for the logic
        self.data = ShotsData(radius=15.0)
        
        # Tkinter Variables
        self.trials_var = tk.StringVar()
        self.hits_var = tk.StringVar()
        self.radius_var = tk.StringVar(value=str(self.data.valid_radius))
        
        # Create frames
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

        # Treeview for shots
        self.create_shot_entries()

        # Add & Remove
        self.add_button = tk.Button(self.controls_frame, text="Add Shot", command=self.add_shot)
        self.add_button.grid(row=0, column=0, padx=5)

        self.remove_button = tk.Button(self.controls_frame, text="Remove Selected Shot", command=self.remove_shot)
        self.remove_button.grid(row=0, column=1, padx=5)

        # Import/Export
        self.import_button = tk.Button(self.controls_frame, text="Import from Excel", command=self.import_from_excel)
        self.import_button.grid(row=0, column=2, padx=5)

        self.export_button = tk.Button(self.controls_frame, text="Export to Excel", command=self.export_to_excel)
        self.export_button.grid(row=0, column=3, padx=5)

        # Radius
        self.radius_label = tk.Label(self.controls_frame, text="Target Radius (cm):")
        self.radius_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.radius_entry = tk.Entry(self.controls_frame, textvariable=self.radius_var, width=10)
        self.radius_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Metrics
        self.avg_label = tk.Label(self.metrics_frame, text="Average Coordinates: N/A")
        self.avg_label.grid(row=0, column=0, sticky='w', padx=10)

        self.accuracy_label = tk.Label(self.metrics_frame, text=f"Accuracy (within {self.data.valid_radius}cm): N/A")
        self.accuracy_label.grid(row=1, column=0, sticky='w', padx=10)

        self.stdX_label = tk.Label(self.metrics_frame, text="Standard Deviation X: N/A")
        self.stdX_label.grid(row=2, column=0, sticky='w', padx=10)

        self.stdY_label = tk.Label(self.metrics_frame, text="Standard Deviation Y: N/A")
        self.stdY_label.grid(row=3, column=0, sticky='w', padx=10)

        self.rms_x_label = tk.Label(self.metrics_frame, text="RMS X: N/A")
        self.rms_x_label.grid(row=4, column=0, sticky='w', padx=10)

        self.rms_y_label = tk.Label(self.metrics_frame, text="RMS Y: N/A")
        self.rms_y_label.grid(row=5, column=0, sticky='w', padx=10)

        self.rms_radial_label = tk.Label(self.metrics_frame, text="RMS Radial: N/A")
        self.rms_radial_label.grid(row=6, column=0, sticky='w', padx=10)

        self.range_x_50_label = tk.Label(self.metrics_frame, text="50% X Range: N/A")
        self.range_x_50_label.grid(row=7, column=0, sticky='w', padx=10)

        self.range_y_50_label = tk.Label(self.metrics_frame, text="50% Y Range: N/A")
        self.range_y_50_label.grid(row=8, column=0, sticky='w', padx=10)

        self.cep_50_label = tk.Label(self.metrics_frame, text="CEP50 (radius): N/A")
        self.cep_50_label.grid(row=9, column=0, sticky='w', padx=10)

        self.cumulative_x_error_label = tk.Label(self.metrics_frame, text="Average absolute X: N/A")
        self.cumulative_x_error_label.grid(row=10, column=0, sticky='w', padx=10)

        self.cumulative_y_error_label = tk.Label(self.metrics_frame, text="Average absolute Y: N/A")
        self.cumulative_y_error_label.grid(row=11, column=0, sticky='w', padx=10)

        self.cep_100_label = tk.Label(self.metrics_frame, text="CEP100 (radius): N/A")
        self.cep_100_label.grid(row=12, column=0, sticky='w', padx=10)

        self.range_x_100_label = tk.Label(self.metrics_frame, text="100% X Range: N/A")
        self.range_x_100_label.grid(row=13, column=0, sticky='w', padx=10)

        self.range_y_100_label = tk.Label(self.metrics_frame, text="100% Y Range: N/A")
        self.range_y_100_label.grid(row=14, column=0, sticky='w', padx=10)

        # Probability area
        self.trials_label = tk.Label(self.prob_frame, text="Number of Trials:")
        self.trials_label.grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.trials_entry = tk.Entry(self.prob_frame, textvariable=self.trials_var)
        self.trials_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.hits_label = tk.Label(self.prob_frame, text="Number of Hits Within Radius:")
        self.hits_label.grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.hits_entry = tk.Entry(self.prob_frame, textvariable=self.hits_var)
        self.hits_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.prob_xy_label = tk.Label(self.prob_frame, text="Probability of one shot hitting: N/A")
        self.prob_xy_label.grid(row=6, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_binomial_label = tk.Label(self.prob_frame, text="Probability of reaching desired result: N/A")
        self.prob_binomial_label.grid(row=7, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_lower_label = tk.Label(self.prob_frame, text="Lower Probability (95% confidence): N/A")
        self.prob_lower_label.grid(row=8, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_higher_label = tk.Label(self.prob_frame, text="Higher Probability (95% confidence): N/A")
        self.prob_higher_label.grid(row=9, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_lower_50_label = tk.Label(self.prob_frame, text="Lower Probability (50% confidence): N/A")
        self.prob_lower_50_label.grid(row=10, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_higher_50_label = tk.Label(self.prob_frame, text="Higher Probability (50% confidence): N/A")
        self.prob_higher_50_label.grid(row=11, column=0, columnspan=2, sticky='w', padx=10)

        # Event bindings
        self.trials_var.trace_add('write', self.on_prob_input_change)
        self.hits_var.trace_add('write', self.on_prob_input_change)
        self.radius_var.trace_add('write', self.on_radius_change)

        # Plots
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
            x_str = x_entry.get().strip()
            y_str = y_entry.get().strip()
            if x_str == "" or y_str == "":
                messagebox.showerror("Input Error", "Both X and Y coordinates must be filled.")
                return
            try:
                x_val = float(x_str)
                y_val = float(y_str)
                # Add to data
                self.data.add_shot(x_val, y_val)
                # Add to tree
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
            index = self.tree.index(item)
            self.tree.delete(item)
            self.data.remove_shot(index)
        self.update_metrics_and_visualization()

    def import_from_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            self.data.import_from_excel(file_path)
            # Clear tree
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Repopulate
            for x, y in self.data.shots:
                self.tree.insert('', 'end', values=(x, y))
            self.update_metrics_and_visualization()
            messagebox.showinfo("Import Successful", f"Data imported successfully from {file_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"An error occurred while importing: {e}")

    def export_to_excel(self):
        if not self.data.shots:
            messagebox.showerror("No Data", "No data to export.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension='.xlsx',
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.data.export_to_excel(file_path)
            messagebox.showinfo("Export Successful", f"Data exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting: {e}")

    def on_shot_change(self, event):
        """
        Whenever user edits the tree directly (though your code doesn’t show
        cell-edit in place, you might have a plugin or future extension),
        re-sync ShotsData from tree. Then recalc.
        """
        # Rebuild shots from the tree
        new_shots = []
        for item in self.tree.get_children():
            x, y = self.tree.item(item)['values']
            new_shots.append((float(x), float(y)))
        self.data.shots = new_shots
        self.update_metrics_and_visualization()

    def update_metrics_and_visualization(self):
        # 1) Recalculate metrics
        self.data.calculate_metrics()
        # 2) Probability calculations
        self.data.calculate_probabilities()
        # 3) Update metric labels
        self.update_metric_labels()
        # 4) Update plots
        self.update_visualization()

    def update_metric_labels(self):
        if not self.data.shots:
            # Clear everything
            self.avg_label.config(text="Average Coordinates: N/A")
            self.accuracy_label.config(text=f"Accuracy (within {self.data.valid_radius}cm): N/A")
            self.stdX_label.config(text="Standard Deviation X: N/A")
            self.stdY_label.config(text="Standard Deviation Y: N/A")
            self.rms_x_label.config(text="RMS X: N/A")
            self.rms_y_label.config(text="RMS Y: N/A")
            self.rms_radial_label.config(text="RMS Radial: N/A")
            self.range_x_50_label.config(text="50% X Range: N/A")
            self.range_y_50_label.config(text="50% Y Range: N/A")
            self.cep_50_label.config(text="CEP50 (radius): N/A")
            self.cumulative_x_error_label.config(text="Average absolute X: N/A")
            self.cumulative_y_error_label.config(text="Average absolute Y: N/A")
            self.cep_100_label.config(text="CEP100 (radius): N/A")
            self.range_x_100_label.config(text="100% X Range: N/A")
            self.range_y_100_label.config(text="100% Y Range: N/A")
            return
        
        # Basic
        avg_x, avg_y = self.data.avg_coords
        self.avg_label.config(text=f"Average Coordinates: ({avg_x:.2f}, {avg_y:.2f}) cm")
        within_radius = self.data.accuracy
        total = self.data.total_shots
        self.accuracy_label.config(
            text=f"Accuracy (within {self.data.valid_radius}cm): {within_radius} / {total}"
        )
        
        if self.data.stdX_dev is not None:
            self.stdX_label.config(text=f"Standard Deviation X: {self.data.stdX_dev:.2f} cm")
            self.stdY_label.config(text=f"Standard Deviation Y: {self.data.stdY_dev:.2f} cm")
        else:
            self.stdX_label.config(text="Standard Deviation X: N/A")
            self.stdY_label.config(text="Standard Deviation Y: N/A")
        
        # If 2+ shots, we can do extra stats
        distances = self.data.distances
        if len(distances) > 1:
            x_vals = [s[0] for s in self.data.shots]
            y_vals = [s[1] for s in self.data.shots]
            # RMS X
            rms_x = np.sqrt(np.mean((np.array(x_vals) - avg_x)**2))
            rms_y = np.sqrt(np.mean((np.array(y_vals) - avg_y)**2))
            rms_radial = np.sqrt(np.mean(np.array(distances)**2))
            self.rms_x_label.config(text=f"RMS X: {rms_x:.2f} cm")
            self.rms_y_label.config(text=f"RMS Y: {rms_y:.2f} cm")
            self.rms_radial_label.config(text=f"RMS Radial: {rms_radial:.2f} cm")
            
            x_50_range = 2 * np.percentile(np.abs(np.array(x_vals) - avg_x), 50)
            y_50_range = 2 * np.percentile(np.abs(np.array(y_vals) - avg_y), 50)
            self.range_x_50_label.config(text=f"50% X Range: {x_50_range:.2f} cm")
            self.range_y_50_label.config(text=f"50% Y Range: {y_50_range:.2f} cm")
            
            cep_50 = np.percentile(distances, 50)
            self.cep_50_label.config(text=f"CEP50 (radius): {cep_50:.2f} cm")
            
            cum_x_error = float(np.sum(np.abs(np.array(x_vals) - avg_x)))
            cum_y_error = float(np.sum(np.abs(np.array(y_vals) - avg_y)))
            n = len(distances)
            self.cumulative_x_error_label.config(text=f"Average absolute X: {cum_x_error/n:.2f} cm")
            self.cumulative_y_error_label.config(text=f"Average absolute Y: {cum_y_error/n:.2f} cm")
            
            cep_100 = max(distances)
            range_x_100 = max(x_vals) - min(x_vals)
            range_y_100 = max(y_vals) - min(y_vals)
            self.cep_100_label.config(text=f"CEP100 (radius): {cep_100:.2f} cm")
            self.range_x_100_label.config(text=f"100% X Range: {range_x_100:.2f} cm")
            self.range_y_100_label.config(text=f"100% Y Range: {range_y_100:.2f} cm")
        else:
            # If only 1 shot
            self.rms_x_label.config(text="RMS X: N/A")
            self.rms_y_label.config(text="RMS Y: N/A")
            self.rms_radial_label.config(text="RMS Radial: N/A")
            self.range_x_50_label.config(text="50% X Range: N/A")
            self.range_y_50_label.config(text="50% Y Range: N/A")
            self.cep_50_label.config(text="CEP50 (radius): N/A")
            self.cumulative_x_error_label.config(text="Average absolute X: N/A")
            self.cumulative_y_error_label.config(text="Average absolute Y: N/A")
            self.cep_100_label.config(text="CEP100 (radius): N/A")
            self.range_x_100_label.config(text="100% X Range: N/A")
            self.range_y_100_label.config(text="100% Y Range: N/A")
        
        # Probability
        if self.data.prob_hit_one_shot is not None:
            self.prob_xy_label.config(
                text=f"Probability of one shot hitting: {self.data.prob_hit_one_shot * 100:.2f}%"
            )
        else:
            self.prob_xy_label.config(text="Probability of one shot hitting: N/A")
        
        if self.data.prob_binomial is not None:
            self.prob_binomial_label.config(
                text=f"Probability of reaching desired result: {self.data.prob_binomial * 100:.2f}%"
            )
            self.prob_lower_label.config(
                text=f"Lower Probability (95% confidence): {self.data.prob_binomial_lower_95 * 100:.2f}%"
            )
            self.prob_higher_label.config(
                text=f"Higher Probability (95% confidence): {self.data.prob_binomial_higher_95 * 100:.2f}%"
            )
            self.prob_lower_50_label.config(
                text=f"Lower Probability (50% confidence): {self.data.prob_binomial_lower_50 * 100:.2f}%"
            )
            self.prob_higher_50_label.config(
                text=f"Higher Probability (50% confidence): {self.data.prob_binomial_higher_50 * 100:.2f}%"
            )
        else:
            self.prob_binomial_label.config(text="Probability of reaching desired result: N/A")
            self.prob_lower_label.config(text="Lower Probability (95% confidence): N/A")
            self.prob_higher_label.config(text="Higher Probability (95% confidence): N/A")
            self.prob_lower_50_label.config(text="Lower Probability (50% confidence): N/A")
            self.prob_higher_50_label.config(text="Higher Probability (50% confidence): N/A")

    def on_radius_change(self, *args):
        try:
            new_radius = float(self.radius_var.get())
            self.data.set_radius(new_radius)
            self.update_metrics_and_visualization()
        except ValueError:
            # revert
            self.radius_var.set(str(self.data.valid_radius))

    def on_prob_input_change(self, *args):
        # parse the user’s typed values for trials/hits
        try:
            t = int(self.trials_var.get().strip())
        except:
            t = None
        try:
            h = int(self.hits_var.get().strip())
        except:
            h = None
        
        if t is not None:
            try:
                self.data.set_trials(t)
            except ValueError:
                pass
        if h is not None:
            try:
                self.data.set_hits(h)
            except ValueError:
                pass
        
        self.data.calculate_probabilities()
        self.update_metric_labels()

    def setup_visualization_plot(self):
        self.visualization_frame = tk.Frame(self.plots_frame)
        self.visualization_frame.pack(side='left', fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.density_frame = tk.Frame(self.plots_frame)
        self.density_frame.pack(side='right', fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig_vis, self.ax_vis = plt.subplots(figsize=(6,6))
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.visualization_frame)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)
        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        self.fig_density, (self.ax_density_x, self.ax_density_y) = plt.subplots(2, 1, figsize=(6,6))
        self.canvas_density = FigureCanvasTkAgg(self.fig_density, master=self.density_frame)
        self.canvas_density.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
        # Distribution plot
        self.ax_vis.clear()
        self.ax_vis.set_title('Shot Distribution')
        self.ax_vis.set_xlabel('X (cm)')
        self.ax_vis.set_ylabel('Y (cm)')
        self.ax_vis.grid(True)

        shots = self.data.shots
        if shots:
            x_vals = [s[0] for s in shots]
            y_vals = [s[1] for s in shots]
            self.ax_vis.scatter(x_vals, y_vals, c='blue', label='Shots')

            if self.data.avg_coords:
                avg_x, avg_y = self.data.avg_coords
                self.ax_vis.scatter(avg_x, avg_y, c='green', marker='x', s=100, label='Center')
                circle = Circle((avg_x, avg_y), self.data.valid_radius, color='red',
                                fill=False, linestyle='--',
                                label=f'{self.data.valid_radius}cm Radius')
                self.ax_vis.add_patch(circle)

            self.ax_vis.legend()
            self.ax_vis.set_aspect('equal', adjustable='datalim')

        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        # Density plots (X & Y)
        self.ax_density_x.clear()
        self.ax_density_y.clear()

        if len(shots) > 2:
            from scipy.stats import gaussian_kde
            x_vals = [s[0] for s in shots]
            y_vals = [s[1] for s in shots]

            kde_x = gaussian_kde(x_vals, bw_method='scott')
            x_range = np.linspace(min(x_vals) - 10, max(x_vals) + 10, 1000)
            y_kde_x = kde_x(x_range)
            self.ax_density_x.plot(x_range, y_kde_x, color='blue', lw=2)
            self.ax_density_x.fill_between(x_range, y_kde_x, color='skyblue', alpha=0.5)
            self.ax_density_x.set_title('X Coordinate Density')
            self.ax_density_x.set_xlabel('X (cm)')
            self.ax_density_x.set_ylabel('Density')
            self.ax_density_x.grid(True)

            kde_y = gaussian_kde(y_vals, bw_method='scott')
            y_range = np.linspace(min(y_vals) - 10, max(y_vals) + 10, 1000)
            y_kde_y = kde_y(y_range)
            self.ax_density_y.plot(y_range, y_kde_y, color='blue', lw=2)
            self.ax_density_y.fill_between(y_range, y_kde_y, color='skyblue', alpha=0.5)
            self.ax_density_y.set_title('Y Coordinate Density')
            self.ax_density_y.set_xlabel('Y (cm)')
            self.ax_density_y.set_ylabel('Density')
            self.ax_density_y.grid(True)
        else:
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

###############################################################################
# 3) CLI mode: text-based commands to do the same add/list/remove/import/export
###############################################################################

def cli_mode():
    print("Entering CLI mode. Type 'help' for commands, 'exit' to quit.")
    data = ShotsData(radius=15.0)

    def do_help():
        print("Available commands:")
        print("  help                     Show this help")
        print("  list                     Show all shots")
        print("  add <x> <y>             Add a shot")
        print("  remove <index>          Remove a shot by its index (from 'list')")
        print("  import <file.xlsx>      Import from Excel")
        print("  export <file.xlsx>      Export to Excel")
        print("  set radius <value>      Set radius (cm)")
        print("  set trials <N>          Set number of trials")
        print("  set hits <N>            Set number of hits")
        print("  calc                    Calculate all metrics & probabilities")
        print("  metrics                 Print out the latest metrics")
        print("  exit                    Quit the program")

    def do_list():
        shots = data.list_shots()
        if not shots:
            print("No shots.")
        else:
            for i, x, y in shots:
                print(f"{i}: X={x}, Y={y}")
    
    def do_add(args):
        if len(args) < 2:
            print("Usage: add <x> <y>")
            return
        try:
            x = float(args[0])
            y = float(args[1])
            data.add_shot(x, y)
            print(f"Shot added: (X={x}, Y={y})")
        except ValueError:
            print("Invalid numeric values.")
    
    def do_remove(args):
        if not args:
            print("Usage: remove <index>")
            return
        try:
            index = int(args[0])
            data.remove_shot(index)
            print(f"Removed shot at index {index}")
        except ValueError:
            print("Index must be an integer.")
    
    def do_import(args):
        if not args:
            print("Usage: import <file.xlsx>")
            return
        path = " ".join(args)
        try:
            data.import_from_excel(path)
            print(f"Imported shots from {path}")
        except Exception as e:
            print(f"Import error: {e}")
    
    def do_export(args):
        if not args:
            print("Usage: export <file.xlsx>")
            return
        path = " ".join(args)
        try:
            data.export_to_excel(path)
            print(f"Exported shots to {path}")
        except Exception as e:
            print(f"Export error: {e}")
    
    def do_set(args):
        if len(args) < 2:
            print("Usage: set radius/trials/hits <value>")
            return
        field = args[0]
        value = args[1]
        try:
            val = float(value)
        except:
            print("Value must be numeric.")
            return
        if field == "radius":
            try:
                data.set_radius(val)
                print(f"Radius set to {val}")
            except ValueError as e:
                print(e)
        elif field == "trials":
            try:
                ival = int(value)
                data.set_trials(ival)
                print(f"Trials set to {ival}")
            except ValueError as e:
                print(e)
        elif field == "hits":
            try:
                ival = int(value)
                data.set_hits(ival)
                print(f"Hits set to {ival}")
            except ValueError as e:
                print(e)
        else:
            print(f"Unknown field '{field}'.")
    
    def do_calc():
        data.calculate_metrics()
        data.calculate_probabilities()
        print("Metrics & probabilities recalculated. Use 'metrics' to view.")
    
    def do_metrics():
        """Print out the main metrics from ShotsData."""
        data.calculate_metrics()  # in case not done
        data.calculate_probabilities()
        
        if not data.shots:
            print("No shots to show metrics for.")
            return
        
        avg_coords = data.avg_coords
        print(f"Average coords: {avg_coords}")
        print(f"Shots within radius {data.valid_radius}cm: {data.accuracy} / {data.total_shots}")
        print(f"Std dev X={data.stdX_dev}, Y={data.stdY_dev}")
        
        # We won’t replicate absolutely all, but you can adapt as needed
        if data.prob_hit_one_shot is not None:
            print(f"Probability of 1 shot hitting: {data.prob_hit_one_shot*100:.2f}%")
        if data.prob_binomial is not None:
            print(f"Probability of reaching desired result (binomial): {data.prob_binomial*100:.2f}%")
            print(f"95% CI: [{data.prob_binomial_lower_95*100:.2f}%, {data.prob_binomial_higher_95*100:.2f}%]")
            print(f"50% CI: [{data.prob_binomial_lower_50*100:.2f}%, {data.prob_binomial_higher_50*100:.2f}%]")

    do_help()
    while True:
        try:
            line = input("cli> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == 'exit':
            print("Goodbye!")
            break
        elif cmd == 'help':
            do_help()
        elif cmd == 'list':
            do_list()
        elif cmd == 'add':
            do_add(args)
        elif cmd == 'remove':
            do_remove(args)
        elif cmd == 'import':
            do_import(args)
        elif cmd == 'export':
            do_export(args)
        elif cmd == 'set':
            do_set(args)
        elif cmd == 'calc':
            do_calc()
        elif cmd == 'metrics':
            do_metrics()
        else:
            print(f"Unknown command: {cmd}. Type 'help' for a list.")

###############################################################################
# 4) main(): Decide whether to run the GUI or CLI based on arguments
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run the GUI instead of CLI")
    args = parser.parse_args()
    
    if args.gui:
        root = tk.Tk()
        root.state('zoomed')
        app = ShotAccuracyApp(root)
        root.mainloop()
    else:
        cli_mode()

if __name__ == "__main__":
    main()
