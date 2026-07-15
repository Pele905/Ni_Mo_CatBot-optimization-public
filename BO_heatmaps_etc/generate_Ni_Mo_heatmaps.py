import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.gridspec import GridSpec

# Import helper functions and globals.
# They are defined in your plotting_helpers file:
#   - get_torch_from_df: converts a pandas DataFrame to a torch tensor following variable_order.
#   - get_my_gp: returns the final GP (posterior) model.
#   - variable_names: a mapping from original names to new names.
#   - variable_order: the list of input variable names in a prescribed order.
#   - human_names: a mapping of variable names to human‐readable labels.
#   - bounds: an array (or list) with each row giving the [min, max] for the variable.
from plotting_helpers import (
    get_torch_from_df,
    get_my_gp,
    variable_names,
    variable_order,
    human_names,
    bounds
)

# ------------------------ Data and Model Setup ------------------------------

# Load CSV data (change the path if necessary)
data = pd.read_csv("/Users/pvifr/Desktop/ElectrochemicalDataAnalysis/Ni_Mo_CatBot optimization public/complete_dataset.csv", delimiter=",")
data = data.rename(columns=variable_names)
print(data)
ycol = "stability_slope"
# Reorder columns according to variable_order (plus y and 'experiment' if available)
data = data[variable_order + [ycol]]

# Convert data to torch tensors for the GP model fitting.
xdatatensor = get_torch_from_df(data, variable_order)
ydatatensor = torch.tensor(data[ycol].values, dtype=torch.double).unsqueeze(-1)
best_gp = get_my_gp(xdatatensor, ydatatensor)

# Choose the best data point (e.g., the one with the lowest stability_slope)
best_data = data.sort_values(by=ycol, ascending=True)
best_xdata_tensor = get_torch_from_df(best_data, variable_order)
# Use the first row (i.e., the best) as the default base input vector.
base_input_default = best_xdata_tensor[0].clone()
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def generate_heatmap_row(variable_order_dict, variable_order_set_values, sweep_var, sweep_values):
    """
    sweep_var: str, the variable to sweep across subplots (e.g., "liquid1", "temperature")
    sweep_values: list of float, values for that variable
    """
    var_y = variable_order_dict["target_variable_y"]
    var_x = variable_order_dict["target_variable_x"]

    idx_y = variable_order.index(var_y)
    idx_x = variable_order.index(var_x)

    fixed_vars = [var for var in variable_order if var not in [var_x, var_y]]

    ncols = len(sweep_values)
    fig = plt.figure(figsize=(6*ncols, 6))
    gs = GridSpec(1, ncols+1, figure=fig, width_ratios=[1]*ncols + [0.05])
    axs = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    cax = fig.add_subplot(gs[0, ncols])  # colorbar axis

    # compute heatmaps and store means for shared color limits
    means_list = []
    all_means = []

    npoints = 20
    x_vals = torch.linspace(bounds[idx_x, 0] if isinstance(bounds, np.ndarray) else bounds[idx_x][0],
                            bounds[idx_x, 1] if isinstance(bounds, np.ndarray) else bounds[idx_x][1],
                            npoints)
    y_vals = torch.linspace(bounds[idx_y, 0] if isinstance(bounds, np.ndarray) else bounds[idx_y][0],
                            bounds[idx_y, 1] if isinstance(bounds, np.ndarray) else bounds[idx_y][1],
                            npoints)

    for val in sweep_values:
        # start from defaults
        fixed_values = [base_input_default[variable_order.index(var)].item() for var in fixed_vars]
        if sweep_var in fixed_vars:
            idx = fixed_vars.index(sweep_var)
            fixed_values[idx] = val

        means = np.zeros((npoints, npoints))
        for i, yv in enumerate(y_vals):
            for j, xv in enumerate(x_vals):
                inp = base_input_default.clone()
                inp[idx_x] = xv
                inp[idx_y] = yv
                for var, fv in zip(fixed_vars, fixed_values):
                    inp[variable_order.index(var)] = fv
                posterior = best_gp.posterior(inp.unsqueeze(0))
                means[i, j] = posterior.mean.item()
        means_list.append(means)
        all_means.append(means)

    # shared color limits
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)

    # plot each heatmap
    for ax, val, means in zip(axs, sweep_values, means_list):
        X, Y = np.meshgrid(x_vals, y_vals)
        c = ax.pcolormesh(X, Y, means, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel(human_names[var_x], fontsize=20)
        if ax == axs[0]:
            ax.set_ylabel(human_names[var_y], fontsize=20)
        ax.set_box_aspect(1)  # square axes
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(f'{sweep_var} = {val}', fontsize=18)

    # shared colorbar
    cbar = fig.colorbar(c, cax=cax)
    cbar.set_label(r'$\langle \eta_{-10} \rangle$', fontsize=35)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()

variable_order_dict = {"target_variable_y": "liquid2", 
                       "target_variable_x": "H2SO4"}

variable_order_set_vals = {"temperature": 60, 
                           "liquid1": 0.3,  
                           "current_density" : 100, 
                           "deposition_time" : 50
                           }

# Sweep NiSO4
#generate_heatmap_row(variable_order_dict, variable_order_set_vals, sweep_var="liquid1", sweep_values=[0.1, 0.25, 0.35])

# Sweep Temperature
#generate_heatmap_row(variable_order_dict, variable_order_set_vals, sweep_var="temperature", sweep_values=[40, 50, 60])
import torch
import numpy as np
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- User settings --------------------
temperatures = [40, 50, 60]

# Example variable grids
Na2Mo_list = np.linspace(0.015, 0.15, 25)
NiSO4_list = np.linspace(0.015, 0.35, 25)
H2SO4_list = np.linspace(0.03, 0.1, 6)

dep_time_fixed = 350
dep_current_density_fixed = 70

# General axis assignment: change these to swap variables
axis_vars = {
    'x': 'liquid1',   # e.g., NiSO4
    'y': 'liquid2',
    'z': 'H2SO4',     # e.g., Na2Mo
    'temperature': 'temperature',
    'dep_time': 'deposition_time',
    'I': 'current_density'
}

idx_x = variable_order.index(axis_vars['x'])
idx_y = variable_order.index(axis_vars['y'])
idx_z = variable_order.index(axis_vars['z'])
idx_temp = variable_order.index(axis_vars['temperature'])
idx_dep_t = variable_order.index(axis_vars['dep_time'])
idx_I = variable_order.index(axis_vars['I'])

# Map axis variables to their arrays
var_ranges = {
    'Na2Mo': Na2Mo_list,
    'NiSO4': NiSO4_list,
    'H2SO4': H2SO4_list,
    'liquid1': NiSO4_list,   # if your variable_order uses these names
    'liquid2': Na2Mo_list,
    'temperature': temperatures
}

# -------------------- Create subplots --------------------
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type':'scene'}, {'type':'scene'}, {'type':'scene'}]],
    subplot_titles=[f"T = {T} °C" for T in temperatures],
    horizontal_spacing=0.01
)
# Lower titles slightly and adjust font size
for annotation in fig.layout.annotations:
    annotation.y = 0.83
    annotation.font.size = 30
    annotation.font.family = "Arial"

# -------------------- Compute predictions --------------------
all_Z_colors = []
for T in temperatures:
    for z_val in var_ranges[axis_vars['z']]:  # use chosen z variable
        Z_color = np.zeros((len(var_ranges[axis_vars['y']]), len(var_ranges[axis_vars['x']])))
        for i, y_val in enumerate(var_ranges[axis_vars['y']]):
            for j, x_val in enumerate(var_ranges[axis_vars['x']]):
                inp = base_input_default.clone()
                inp[idx_x] = x_val
                inp[idx_y] = y_val
                inp[idx_z] = z_val
                inp[idx_temp] = T
                inp[idx_dep_t] = dep_time_fixed
                inp[idx_I] = dep_current_density_fixed

                posterior = best_gp.posterior(inp.unsqueeze(0))
                Z_color[i, j] = min(posterior.mean.item(), 30000)  # cap posterior
        all_Z_colors.append(Z_color)

# Global color limits for consistent coloring
vmin = min(Z.min() for Z in all_Z_colors)
vmax = max(Z.max() for Z in all_Z_colors)
if vmin < 5000:
    print(vmin)
    
print(vmin, vmax, "These are maxes and minimums")
# -------------------- Add surfaces --------------------
for col, T in enumerate(temperatures):
    for z_val in var_ranges[axis_vars['z']]:  # use chosen z variable
        Z_color = np.zeros((len(var_ranges[axis_vars['y']]), len(var_ranges[axis_vars['x']])))
        for i, y_val in enumerate(var_ranges[axis_vars['y']]):
            for j, x_val in enumerate(var_ranges[axis_vars['x']]):
                inp = base_input_default.clone()
                inp[idx_x] = x_val
                inp[idx_y] = y_val
                inp[idx_z] = z_val
                inp[idx_temp] = T
                inp[idx_dep_t] = dep_time_fixed
                inp[idx_I] = dep_current_density_fixed

                posterior = best_gp.posterior(inp.unsqueeze(0))
                Z_color[i, j] = min(posterior.mean.item(), 30000)

        X, Y = np.meshgrid(var_ranges[axis_vars['x']], var_ranges[axis_vars['y']])
        Z_height = np.ones_like(X) * z_val

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_height,
                surfacecolor=Z_color,
                colorscale='RdBu',
                cmin=vmin,
                cmax=vmax,
                showscale=False if not (col == len(temperatures)-1 and z_val == var_ranges[axis_vars['z']][-1]) else True,
                name=f"{axis_vars['z']}={z_val:.2f}",
                colorbar=dict(
                    len=0.7,
                    y=0.5,
                    thickness=20,
                    tickvals=[vmin, vmax],
                    ticktext=[f"{vmin / 100:.0f}", f"{vmax / 100:.0f}"],
                    tickfont=dict(size=20),   # make ticks larger
                    x=0.98,                   # move closer to the plot (lower = closer)
                    title=dict(
                        text="",               # leave empty, we’ll add annotation for horizontal title
                        side="top",
                        font=dict(size=30),
                    )
                )
            ),
            row=1, col=col+1
        )

fig.add_annotation(
    x=1.01, y=0.9,  # above colorbar
    xref='paper', yref='paper',
    text=r'$\langle \eta_{-10} \rangle$', 
    showarrow=False,
    font=dict(
        size=34,
        family="Arial, Bold",   # bold font
        color="black"
    )
)

# -------------------- Layout --------------------
zoom = 2
scenes = ['scene', 'scene2', 'scene3']
for sc in scenes:
    fig.update_layout(
        **{
            sc: dict(
                xaxis=dict(
                    title="", #"NiSO₄ [M]",
                    title_font=dict(size=24),
                    tickfont=dict(size=19)
                ),
                yaxis=dict(
                    title="", #"Na₂MoO₄ [M]",
                    title_font=dict(size=24),
                    tickfont=dict(size=19)
                ),
                zaxis=dict(
                    title="", #"H₂SO₄ [M]",
                    title_font=dict(size=24),
                    tickfont=dict(size=19)
                ),
                aspectmode='cube',
                camera=dict(eye=dict(x=zoom, y=zoom, z=zoom))
            )
        }
    )
#fig.write_image("heatmap.png", width=1800, height=600)
fig.write_html("figure.html")
fig.show(renderer="browser")
