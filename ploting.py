import numpy as np
import plotly.graph_objects as go

def test_plot(cloud):
    # Assuming your point cloud is stored in the 'cloud' variable
    #cloud = np.random.rand(100, 3)  # Replace with your actual data

    # Define the indices of the points you want to color red
    red_indices = np.random.choice(range(cloud.shape[0]), 10, replace=False)

    # Create a figure
    fig = go.Figure(data=go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            line=dict(color='black', width=1),
            color=['blue' if i not in red_indices else 'red' for i in range(cloud.shape[0])]
        )
    ))

    # Show the figure
    fig.show()


def test_plot2(cloud, special_indices=None):
    # Assuming your point cloud is stored in the 'cloud' variable
    # Define the indices of the points you want to color red
    if special_indices is None:
        special_indices = np.random.choice(range(cloud.shape[0]), 30, replace=False)



    # Create a figure
    fig = go.Figure(data=go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=[8 if i in special_indices else 2 for i in range(cloud.shape[0])],
            line=dict(color='black', width=1),
            color=['red' if i in special_indices else 'blue' for i in range(cloud.shape[0])]
        )
    ))

    # Show the figure
    fig.show()


def plot_pc(pc, second_pc=None, s=4, o=0.6):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]], )
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
        row=1, col=1
    )
    if second_pc is not None:
        fig.add_trace(
            go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers',
                         marker=dict(size=s, opacity=o)),
            row=1, col=2
        )
    fig.update_layout(scene_aspectmode='data')
    fig.show()

