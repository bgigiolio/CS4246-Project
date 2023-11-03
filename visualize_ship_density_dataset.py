import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np

def get_index(input:float, lon_input:bool):
    "0.005, 0.0, -180.015311275, 0.0, -0.005, 85.00264793700009)"
    if lon_input:
        return int((input+180.015311275)/0.005)
    else:
        return int((85.00264793700009-input)/0.005)
    
def get_cropped_ship_data(min_lon, max_lon, min_lat, max_lat):
    "read the ship density tif file, crops the tif to that specific region and saves that as a picture"

    #to index:
    min_colon=get_index(min_lon, lon_input=True)
    max_colon=get_index(max_lon, lon_input=True)
    max_row=get_index(min_lat, lon_input=False) #least latitude leads to the highest row index in the density dataset
    min_row=get_index(max_lat, lon_input=False)

    print(min_colon, max_colon)
    file_path = 'ShipDensity_Commercial1.tif'
    raster_data = rasterio.open(file_path)

    window = Window(min_colon, min_row, max_colon-min_colon, max_row-min_row) #column=long, row=lat, offset
    raster_array = raster_data.read(1, window=window)

    metadata = raster_data.meta.copy()
    metadata['width'], metadata['height'] = window.width, window.height
    metadata['transform'] = rasterio.windows.transform(window, raster_data.transform)

    # Write the cropped raster data to a new TIF file and viualize that data
    output_path = 'example_from_density_dataset.tif'
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(raster_array, 1) #the region we are looking at

    Image.MAX_IMAGE_PIXELS = 2448059988
    # Open the TIF file
    image = Image.open(output_path)

    # Save the image as a PNG file with compression
    output_path = 'example_from_density_dataset.png'
    image.save(output_path, format='PNG', compress_level=5)

    return raster_array

if __name__=='__main__':
    get_cropped_ship_data(88.6, 152.9, -12.4, 31.3) #southeast asia should be