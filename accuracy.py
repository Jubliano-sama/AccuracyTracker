import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from math import sqrt
from statistics import mean, stdev
from scipy.stats import binom
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.stats import norm

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
        self.std_dev = None
        self.std_error = None
        self.total_shots = 0

        # Tkinter Variables for live updates
        self.trials_var = tk.StringVar()
        self.hits_var = tk.StringVar()

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

        # Export to Excel Button
        self.export_button = tk.Button(self.controls_frame, text="Export to Excel", command=self.export_to_excel)
        self.export_button.grid(row=0, column=2, padx=5)

        # Metrics Labels
        self.avg_label = tk.Label(self.metrics_frame, text="Average Coordinates: N/A")
        self.avg_label.grid(row=0, column=0, sticky='w', padx=10)

        self.accuracy_label = tk.Label(self.metrics_frame, text="Accuracy (within 15cm): N/A")
        self.accuracy_label.grid(row=1, column=0, sticky='w', padx=10)

        self.std_label = tk.Label(self.metrics_frame, text="Standard Deviation: N/A")
        self.std_label.grid(row=2, column=0, sticky='w', padx=10)

        self.error_label = tk.Label(self.metrics_frame, text="Error in Std Dev: N/A")
        self.error_label.grid(row=3, column=0, sticky='w', padx=10)

        # Separator
        self.separator = ttk.Separator(master, orient='horizontal')
        self.separator.pack(fill='x', pady=10)

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
        self.prob_usual_label = tk.Label(self.prob_frame, text="Usual Probability: N/A")
        self.prob_usual_label.grid(row=2, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_lower_label = tk.Label(self.prob_frame, text="Lower Probability: N/A")
        self.prob_lower_label.grid(row=3, column=0, columnspan=2, sticky='w', padx=10)

        self.prob_higher_label = tk.Label(self.prob_frame, text="Higher Probability: N/A")
        self.prob_higher_label.grid(row=4, column=0, columnspan=2, sticky='w', padx=10)

        # Bind variable changes for live updates
        self.trials_var.trace_add('write', self.on_prob_input_change)
        self.hits_var.trace_add('write', self.on_prob_input_change)

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

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.input_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        # Bind treeview modifications for live updates
        self.tree.bind('<ButtonRelease-1>', self.on_shot_change)
        self.tree.bind('<KeyRelease>', self.on_shot_change)

    def add_shot(self):
        # Open a new window to input X and Y
        add_window = tk.Toplevel(self.master)
        add_window.title("Add New Shot")
        add_window.geometry("300x150")
        add_window.grab_set()  # Make the window modal

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
            self.accuracy_label.config(text="Accuracy (within 15cm): N/A")
            self.std_label.config(text="Standard Deviation: N/A")
            self.error_label.config(text="Error in Std Dev: N/A")
            self.avg_coords = None
            self.distances = []
            self.accuracy = None
            self.std_dev = None
            self.std_error = None
            self.total_shots = 0
            return

        # Calculate average coordinates
        avg_x = mean([shot[0] for shot in shots])
        avg_y = mean([shot[1] for shot in shots])
        self.avg_label.config(text=f"Average Coordinates: ({avg_x:.2f}, {avg_y:.2f}) cm")

        # Calculate distances from center
        distances = [sqrt((shot[0] - avg_x) ** 2 + (shot[1] - avg_y) ** 2) for shot in shots]
        self.distances = distances

        # Calculate accuracy
        within_radius = sum(1 for d in distances if d <= 15)
        self.accuracy = within_radius
        self.total_shots = len(shots)
        self.accuracy_label.config(text=f"Accuracy (within 15cm): {within_radius} / {self.total_shots}")

        # Calculate standard deviation
        if len(distances) >= 1:
            std_dev = stdev(distances) if len(distances) > 1 else None
            self.std_dev = std_dev
            self.std_label.config(text=f"Standard Deviation: {std_dev:.2f} cm" if std_dev is not None else "Standard Deviation: N/A")

            # Calculate standard error of standard deviation
            std_error = std_dev / sqrt(2 * (len(distances) - 1)) if std_dev is not None and len(distances) > 1 else None
            self.std_error = std_error
            self.error_label.config(text=f"Error in Std Dev: {std_error:.2f} cm" if std_error is not None else "Error in Std Dev: N/A")
        else:
            self.std_dev = None
            self.std_error = None
            self.std_label.config(text=f"Standard Deviation: {std_dev:.2f} cm" if len(distances) > 1 else "Standard Deviation: N/A (insufficient data)")
            self.error_label.config(text="Error in Std Dev: N/A")

        # Store average coordinates
        self.avg_coords = (avg_x, avg_y)

    def calculate_probabilities(self):
        if not self.accuracy or self.total_shots == 0:
            self.prob_usual_label.config(text="Usual Probability: N/A")
            self.prob_lower_label.config(text="Lower Probability: N/A")
            self.prob_higher_label.config(text="Higher Probability: N/A")
            return

        # Get user inputs
        trials_str = self.trials_var.get().strip()
        hits_str = self.hits_var.get().strip()
        if trials_str == "" or hits_str == "":
            self.prob_usual_label.config(text="Usual Probability: N/A")
            self.prob_lower_label.config(text="Lower Probability: N/A")
            self.prob_higher_label.config(text="Higher Probability: N/A")
            return
        try:
            trials = int(trials_str)
            hits = int(hits_str)
            if trials <= 0 or hits < 0 or hits > trials:
                raise ValueError
        except ValueError:
            self.prob_usual_label.config(text="Usual Probability: Invalid Input")
            self.prob_lower_label.config(text="Lower Probability: Invalid Input")
            self.prob_higher_label.config(text="Higher Probability: Invalid Input")
            return

        # Calculate probability based on accuracy
        p = self.accuracy / self.total_shots  # Probability of a single hit within radius

        # Usual probability
        prob_usual = 1 - binom.cdf(hits -1, trials, p)

        # Calculate lower and higher probabilities using error in std dev
        if self.std_error is not None and self.avg_coords:
            # Adjust probability based on error
            # This is a simplistic approach; for more accurate uncertainty, more complex statistical methods are required
            p_lower = max(p - (self.std_error / self.avg_distance()), 0)
            p_higher = min(p + (self.std_error / self.avg_distance()), 1)

            prob_lower = 1 - binom.cdf(hits -1, trials, p_lower)
            prob_higher = 1 - binom.cdf(hits -1, trials, p_higher)
        else:
            prob_lower = prob_higher = None

        # Update labels
        self.prob_usual_label.config(text=f"Usual Probability: {prob_usual*100:.2f}%")
        if prob_lower is not None and prob_higher is not None:
            self.prob_lower_label.config(text=f"Lower Probability: {prob_lower*100:.2f}%")
            self.prob_higher_label.config(text=f"Higher Probability: {prob_higher*100:.2f}%")
        else:
            self.prob_lower_label.config(text="Lower Probability: N/A")
            self.prob_higher_label.config(text="Higher Probability: N/A")

    def export_to_excel(self):
        if not self.shots_data_available():
            messagebox.showerror("No Data", "There is no data to export. Please add shots and calculate metrics first.")
            return

        # Ask user for file location
        file_path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        if not file_path:
            return  # User cancelled

        # Prepare data
        data = [(float(self.tree.item(item)['values'][0]), float(self.tree.item(item)['values'][1])) for item in self.tree.get_children()]
        df_shots = pd.DataFrame(data, columns=['X (cm)', 'Y (cm)'])

        # Metrics
        metrics = {
            'Average X (cm)': [self.avg_coords[0]],
            'Average Y (cm)': [self.avg_coords[1]],
            'Accuracy (within 15cm)': [f"{self.accuracy} / {self.total_shots}"],
            'Standard Deviation (cm)': [f"{self.std_dev:.2f}" if self.std_dev else "N/A"],
            'Error in Std Dev (cm)': [f"{self.std_error:.2f}" if self.std_error else "N/A"]
        }
        df_metrics = pd.DataFrame(metrics)

        # Probabilities
        prob_usual = self.prob_usual_label.cget("text").split(": ")[1] if hasattr(self, 'prob_usual_label') else "N/A"
        prob_lower = self.prob_lower_label.cget("text").split(": ")[1] if hasattr(self, 'prob_lower_label') else "N/A"
        prob_higher = self.prob_higher_label.cget("text").split(": ")[1] if hasattr(self, 'prob_higher_label') else "N/A"

        prob_data = {
            'Probability Type': ['Usual Probability', 'Lower Probability', 'Higher Probability'],
            'Value (%)': [prob_usual, prob_lower, prob_higher]
        }
        df_prob = pd.DataFrame(prob_data)

        # Write to Excel
        try:
            with pd.ExcelWriter(file_path) as writer:
                df_shots.to_excel(writer, sheet_name='Shots', index=False)
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                df_prob.to_excel(writer, sheet_name='Probabilities', index=False)
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

        # Distance Density Plot
        self.fig_density, self.ax_density = plt.subplots(figsize=(6,6))
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
        self.ax_density.set_title('Distance to Center Density')
        self.ax_density.set_xlabel('Distance to Center (cm)')
        self.ax_density.set_ylabel('Density')
        self.ax_density.set_xlim(0, 30)
        self.ax_density.grid(True)
        self.fig_density.tight_layout()
        self.canvas_density.draw()

    def update_visualization(self):
        # Update Shot Distribution Plot
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
                # Plot center
                self.ax_vis.scatter(self.avg_coords[0], self.avg_coords[1], c='green', marker='x', s=100, label='Center')

                # Plot 15cm radius
                circle = Circle(self.avg_coords, 15, color='red', fill=False, linestyle='--', label='15cm Radius')
                self.ax_vis.add_patch(circle)

            self.ax_vis.legend()
            self.ax_vis.set_aspect('equal', adjustable='datalim')

        self.fig_vis.tight_layout()
        self.canvas_vis.draw()

        # Update Distance Density Plot
        self.ax_density.clear()
        self.ax_density.set_title('Distance to Center Density')
        self.ax_density.set_xlabel('Distance to Center (cm)')
        self.ax_density.set_ylabel('Density')
        self.ax_density.grid(True)

        if len(self.distances) > 2:
            # Fit and plot the kernel density estimate
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(self.distances, bw_method='scott')
            x = np.linspace(0, 30, 100)
            y = kde(x)
            self.ax_density.plot(x, y, color='blue', lw=2)
            self.ax_density.fill_between(x, y, color='skyblue', alpha=0.5)

        self.fig_density.tight_layout()
        self.canvas_density.draw()

    def avg_distance(self):
        if self.distances:
            return mean(self.distances)
        return 1  # Prevent division by zero

    def on_prob_input_change(self, *args):
        self.calculate_probabilities()

def main():
    # Check for required libraries
    try:
        import tkinter
        import pandas
        import matplotlib
        import scipy
    except ImportError as e:
        missing_module = str(e).split()[-1]
        # Since we can't use messagebox here as it's part of tkinter,
        # we print the error and exit.
        print(f"Missing module: {missing_module}. Please install it using pip.")
        return

    root = tk.Tk()
    app = ShotAccuracyApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()