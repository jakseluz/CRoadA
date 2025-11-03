import numpy as np
from rasterio.transform import from_origin
from rasterio import features

class Rasterizer():

    def get_rasterize_roads(self, gdf_edges, pixel_size = 1):
        min_x, min_y, max_x, max_y = gdf_edges.total_bounds
        width_m = int(max_x - min_x)
        height_m = int(max_y - min_y)

        print(height_m, width_m)
        transform = from_origin(int(min_x), int(max_y), pixel_size, pixel_size)

        grid = features.rasterize(
            [(geometry, 1) for geometry in gdf_edges.geometry],
            out_shape=(height_m, width_m), 
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        print(np.sum(grid)/(grid.shape[0] * grid.shape[1]))

        return grid