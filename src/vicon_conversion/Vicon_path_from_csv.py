import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# --- User Configuration ---
CSV_PATH = r"C:\Python_Projects\RV\rosbag_mid_tower_1.csv"  # Path to your CSV file
colnames = ['Frame','Sub Frame','RX','RY','RZ','TX','TY','TZ']


df = pd.read_csv(
    CSV_PATH,
    skiprows=6,       # zero-based: skip lines 0,1,2,3,4,5
    nrows=26500,
    header=None,      # no header in the remaining file
    names=colnames    # set the column names yourself
)

df2 = pd.read_excel(r"C:\Python_Projects\RV\my_data.xlsx")
print(df.head())
print(df2.head())

x2_list = []
y2_list = []
frames2_list = df2["Unnamed: 0"].astype(float)
tvecs = df2["tvec"]
for tvec in tvecs:
    parts = tvec.strip("[]").split()
    x2, y2, z2 = map(float, parts)
    x2_list.append(x2)
    y2_list.append(z2)


x = df["TX"]
y = df["TY"]
theta = df["RZ"]
frames = df['Frame']

x0 = x.iloc[0]
y0 = y.iloc[0]
x = (x - x0) / 1000
y = (y - y0) / 1000

# Compute unit direction vectors
u = np.cos(theta)
v = np.sin(theta)



# Sample arrows to avoid clutter
SAMPLE_STEP = 10
sample_indices = np.arange(0, len(df), SAMPLE_STEP)

# Build the Plotly figure
fig = go.Figure()

# — plain gray trajectory line —


# — colored markers by frame number —
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(
        size=4,
        color=frames*(15/400),
        colorscale='Viridis',
        colorbar=dict(title='Frame'),
        showscale=True
    ),
    name='Frame'
))

# fig.add_trace(go.Scatter(
#     x=x2_list, y=y2_list,
#     mode='markers',
#     marker=dict(
#         size=4,
#         color=frames2_list,
#         colorscale='Viridis',
#         colorbar=dict(title='Frame'),
#         showscale=True
#     ),
#     name='Frame'
# ))



fig.update_layout(
    title='Top-Down Object Trajectory',
    xaxis_title='X Position [m]',
    yaxis_title='Y Position [m]',
    yaxis_scaleanchor='x',  # equal scaling
    width=700,
    height=700
)

# Show the interactive plot
fig.show()
