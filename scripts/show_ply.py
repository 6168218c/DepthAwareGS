import open3d as o3d
import numpy as np

point_cloud = o3d.io.read_point_cloud("reconstruction.ply")
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
# visualizer.get_render_option().background_color = np.array([0.3, 0.6, 0.3])
visualizer.add_geometry(point_cloud)
visualizer.run()
visualizer.destroy_window()
