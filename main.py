
import os
import sys
import osgeo
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from osgeo import osr
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
# Import biomass specific libraries
# from skimage.morphology import watershed
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestRegressor


def plot_band_array(band_array, image_extent, title, cmap_title, colormap, colormap_limits):
    plt.imshow(band_array, extent=image_extent)
    cbar = plt.colorbar();
    plt.set_cmap(colormap);
    plt.clim(colormap_limits)
    cbar.set_label(cmap_title, rotation=270, labelpad=20)
    plt.title(title);
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    rotatexlabels = plt.setp(ax.get_xticklabels(), rotation=90)


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, epsg):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def raster2array(geotif_file):
    metadata = {}
    dataset = gdal.Open(geotif_file)

    metadata['array_rows'] = dataset.RasterYSize
    metadata['array_cols'] = dataset.RasterXSize
    metadata['bands'] = dataset.RasterCount
    metadata['driver'] = dataset.GetDriver().LongName
    metadata['projection'] = dataset.GetProjection()
    metadata['geotransform'] = dataset.GetGeoTransform()

    mapinfo = dataset.GetGeoTransform()
    metadata['pixelWidth'] = mapinfo[1]
    metadata['pixelHeight'] = mapinfo[5]

    metadata['ext_dict'] = {}
    metadata['ext_dict']['xMin'] = mapinfo[0]
    metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize / mapinfo[1]
    metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize / mapinfo[5]
    metadata['ext_dict']['yMax'] = mapinfo[3]

    metadata['extent'] = (metadata['ext_dict']['xMin'], metadata['ext_dict']['xMax'],
                          metadata['ext_dict']['yMin'], metadata['ext_dict']['yMax'])


    if metadata['bands'] == 1:
        raster = dataset.GetRasterBand(1)
        #print(dataset.GetRasterBand(1).GetScale())
        metadata['noDataValue'] = raster.GetNoDataValue()
        metadata['scaleFactor'] = 1 #raster.GetScale()

        # band statistics
        metadata['bandstats'] = {}  # make a nested dictionary to store band stats in same
        stats = raster.GetStatistics(True, True)
        metadata['bandstats']['min'] = round(stats[0], 2)
        metadata['bandstats']['max'] = round(stats[1], 2)
        metadata['bandstats']['mean'] = round(stats[2], 2)
        metadata['bandstats']['stdev'] = round(stats[3], 2)

        array = dataset.GetRasterBand(1).ReadAsArray(0, 0,
                                                     metadata['array_cols'],
                                                     metadata['array_rows']).astype(np.float)
        array[array == int(metadata['noDataValue'])] = np.nan
        #print(metadata['scaleFactor'])
        array = array / metadata['scaleFactor']
        return array, metadata

    elif metadata['bands'] > 1:
        print('More than one band ... need to modify function for case of multiple bands')


def crown_geometric_volume_pth(tree_data, min_tree_height, pth):
    p = np.percentile(tree_data, pth)
    tree_data_pth = [v if v < p else p for v in tree_data]
    crown_geometric_volume_pth = np.sum(tree_data_pth - min_tree_height)
    return crown_geometric_volume_pth, p


def get_predictors(tree, chm_array, labels):
    indexes_of_tree = np.asarray(np.where(labels == tree.label)).T
    tree_crown_heights = chm_array[indexes_of_tree[:, 0], indexes_of_tree[:, 1]]

    full_crown = np.sum(tree_crown_heights - np.min(tree_crown_heights))

    crown50, p50 = crown_geometric_volume_pth(tree_crown_heights, tree.min_intensity, 50)
    crown60, p60 = crown_geometric_volume_pth(tree_crown_heights, tree.min_intensity, 60)
    crown70, p70 = crown_geometric_volume_pth(tree_crown_heights, tree.min_intensity, 70)

    return [tree.label,
            np.float(tree.area),
            tree.major_axis_length,
            tree.max_intensity,
            tree.min_intensity,
            p50, p60, p70,
            full_crown, crown50, crown60, crown70]


data_path = 'C:/Users/Szymon/Desktop/NEONDATA/'
chm_file = 'NEON_D17_SJER_DP3_256000_4106000_CHM.tif'
'''
shapefile = osgeo.gdal.Open(chm_file)

if shapefile:  # checks to see if shapefile was successfully defined
    numLayers = shapefile.GetLayerCount()
    print(numLayers)
else:  # if it's not successfully defined
    print("L")
'''
just_chm_file = 'NEON_D17_SJER_DP3_256000_4106000_CHM.tif'
just_chm_file_split = just_chm_file.split(sep="_")

chm_array, chm_array_metadata = raster2array('NEON_D17_SJER_DP3_256000_4106000_CHM.tif')

# Plot the original CHM
plt.figure(1)

# Plot the CHM figure
plot_band_array(chm_array, chm_array_metadata['extent'],
                'Canopy height Model',
                'Canopy height (m)',
                'Greens', [0, 9])
plt.savefig(just_chm_file[0:-4] + '_CHM.png', dpi=300, orientation='landscape',
            bbox_inches='tight',
            pad_inches=0.1)


#Smooth the CHM using a gaussian filter to remove spurious points
chm_array_smooth = ndi.gaussian_filter(chm_array,2,
                                       mode='constant',cval=0,truncate=2.0)
chm_array_smooth[chm_array==0] = 0

#Save the smoothed CHM
array2raster('chm_filter.tif',
             (chm_array_metadata['ext_dict']['xMin'],chm_array_metadata['ext_dict']['yMax']),
             1,-1,
             np.array(chm_array_smooth,dtype=float),
             32611)

#Calculate local maximum points in the smoothed CHM

local_maxi = peak_local_max(chm_array_smooth,indices=False, footprint=np.ones((5, 5)))

#Plot the local maximums
plt.figure(2)
plot_band_array(local_maxi.astype(int),chm_array_metadata['extent'],
                'Maximum',
                'Maxi',
                'Greys',
                [0, 1])

plt.savefig(just_chm_file[0:-4]+ '_Maximums.png',
            dpi=300,orientation='landscape',
            bbox_inches='tight',pad_inches=0.1)

array2raster('maximum.tif',
             (chm_array_metadata['ext_dict']['xMin'],chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(local_maxi,dtype=np.float32),32611)
#Identify all the maximum points
markers = ndi.label(local_maxi)[0]



#Create a CHM mask so the segmentation will only occur on the trees
chm_mask = chm_array_smooth
chm_mask[chm_array_smooth != 0] = 1

#Perfrom watershed segmentation
labels = watershed(chm_array_smooth, markers, mask=chm_mask)
labels_for_plot = labels.copy()
labels_for_plot = np.array(labels_for_plot,dtype = np.float32)
labels_for_plot[labels_for_plot==0] = np.nan
max_labels = np.max(labels)

#Plot the segments
plot_band_array(labels_for_plot,chm_array_metadata['extent'],
                'Crown Segmentation','Tree Crown Number',
                'Spectral',[0, max_labels])

plt.savefig(just_chm_file[0:-4]+'_Segmentation.png',
            dpi=300,orientation='landscape',
            bbox_inches='tight',pad_inches=0.1)

array2raster(data_path+'labels.tif',
             (chm_array_metadata['ext_dict']['xMin'],
              chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(labels,dtype=float),32611)

#Get the properties of each segment
tree_properties = regionprops(labels,chm_array)

predictors_chm = np.array([get_predictors(tree, chm_array, labels) for tree in tree_properties])
X = predictors_chm[:,1:]
tree_ids = predictors_chm[:,0]

# Define the file of training data
training_data_file = 'SJER_Biomass_Training.csv'

# Read in the training data from a CSV file
training_data = np.genfromtxt(training_data_file, delimiter=',')

# Grab the biomass (Y) from the first line
biomass = training_data[:, 0]

# Grab the biomass prdeictors from the remaining lines
biomass_predictors = training_data[:, 1:12]

#Define paraemters for Random forest regressor
max_depth = 30

#Define regressor rules
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)

#Fit the biomass to regressor variables
regr_rf.fit(biomass_predictors,biomass)

RandomForestRegressor(max_depth=30, random_state=2)
#Apply the model to the predictors
estimated_biomass = regr_rf.predict(X)

#Set an out raster with the same size as the labels
biomass_map =  np.array((labels),dtype=float)
#Assign the appropriate biomass to the labels
biomass_map[biomass_map==0] = np.nan
for tree_id, biomass_of_tree_id in zip(tree_ids, estimated_biomass):
    biomass_map[biomass_map == tree_id] = biomass_of_tree_id

#Get biomass stats for plotting
mean_biomass = np.mean(estimated_biomass)
std_biomass = np.std(estimated_biomass)
min_biomass = np.min(estimated_biomass)
sum_biomass = np.sum(estimated_biomass)

print('Sum of biomass is ',sum_biomass,' kg')
print('Average biomass is ',mean_biomass,' kg')

#Plot the biomass!
plt.figure(5)
plot_band_array(biomass_map,chm_array_metadata['extent'],
                'Biomass (kg)','Biomass (kg)',
                'winter',
                [min_biomass+std_biomass, mean_biomass+std_biomass*3])

plt.savefig(just_chm_file_split[0]+'_'+just_chm_file_split[1]+'_'+just_chm_file_split[2]+'_'+just_chm_file_split[3]+'_'+just_chm_file_split[4]+'_'+just_chm_file_split[5]+'_'+'Biomass.png',
            dpi=300,orientation='landscape',
            bbox_inches='tight',
            pad_inches=0.1)

array2raster('biomass.tif',
             (chm_array_metadata['ext_dict']['xMin'],chm_array_metadata['ext_dict']['yMax']),
             1,-1,np.array(biomass_map,dtype=float),32611)
