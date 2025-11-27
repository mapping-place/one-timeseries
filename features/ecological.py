import ee
import configparser as cp
import ast
import io
import math
from utils.trees import TreeParser as tp
from features import biophysical

def satelliteEmbeddings(yearStr = None):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    configSatelliteEmbedding = config["FEATURES-SATELLITE-EMBEDDING-V1"]
    satelliteEmbeddingAsset = configSatelliteEmbedding.get('googleSatelliteEmbeddingAnnualV1')

    yearSatelliteEmbedding = ee.ImageCollection(satelliteEmbeddingAsset) \
        .filterDate(yearStr + '-01-01', yearStr + '-12-31') \
        .filterBounds(oneStatesBoundary.geometry()) \
        .mosaic()
    
    return yearSatelliteEmbedding

def assembleAllExistingFeatureRasters(embeddingsYear = None):
    zonesAgroecolNum = biophysical.classificationZonesFromAgroecologicalNumeric(returnExisting = True)
    satelliteEmbedding = satelliteEmbeddings(embeddingsYear)
    lonlat = ee.Image.pixelLonLat().float()
    
    assembled = ee.Image.cat(satelliteEmbedding, zonesAgroecolNum, lonlat)

    return assembled

def assembleFeatureBandsAndExport_yearwiseInFolder(returnExisting = False, year = None, startFreshExport = False, hmmLearnOrClassify = None):
    def addNumericLabelsOfStringLabels(pt):
        l2LabelName = pt.getString("className")
        l2labelNumeric = ee.Dictionary({"labelL2Num": l2labelsDict.getNumber(l2LabelName)})
        l1labelNumeric = ee.Dictionary({"labelL1Num": l2labelsL1Dict.getNumber(l2LabelName)})
        return pt.set(l2labelNumeric.combine(l1labelNumeric))

    def sampleFeaturesGridwise(grids, year):
        def sampleFeatures(grid):
            pointsInGridYr = origTrainingPoints \
                .filter(ee.Filter.eq('year', year)) \
                .filterBounds(grid.geometry())
            satEmbImsInGridYr = ee.ImageCollection(satelliteEmbeddingAsset) \
                .filterBounds(grid.geometry()) \
                .filterDate(year + '-01-01', year + '-02-01') \
                .mosaic()
            zonesAgroecolNum = biophysical.classificationZonesFromAgroecologicalNumeric(returnExisting = True)
            lonlat = ee.Image.pixelLonLat()
            allFeatures = ee.Image.cat(satEmbImsInGridYr, zonesAgroecolNum, lonlat)
            samples = allFeatures.reduceRegions(** {
                "reducer": ee.Reducer.first(),
                "collection": pointsInGridYr,
                "scale": scaleOfExport})
            samplesNumLabelsAdded = samples.map(addNumericLabelsOfStringLabels)
            return samplesNumLabelsAdded

        samplesNumLabelsAdded = grids.map(sampleFeatures).flatten()
        return samplesNumLabelsAdded

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    assetFolder = configCore.get('assetFolderWithFeatures')
    origTrainingPoints = ee.FeatureCollection(configCore.get('lulcLabeledPoints'))
    if hmmLearnOrClassify == "forHMMLearn":
        origTrainingPoints = ee.FeatureCollection(configCore.get('lulcLabeledPointsForLearningTemporalDynamics'))
    scaleOfExport = configCore.getint('scaleOfMap')
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    yearsToPredict = ast.literal_eval(configCore.get("annualTimeseriesYears"))
    configSatelliteEmbedding = config["FEATURES-SATELLITE-EMBEDDING-V1"]
    satelliteEmbeddingAsset = configSatelliteEmbedding.get('googleSatelliteEmbeddingAnnualV1')
    configFeaturesAssemble = config["FEATURES-ASSEMBLE"]
    assetNamePrefix = configFeaturesAssemble.get('existingLabeledPointsWithFeaturesPrefix')
    assetFolderName = configFeaturesAssemble.get('existingLabeledPointsWithFeaturesFolder')
    hmmLearnFolderSuffix = configFeaturesAssemble.get('pointsWithFeaturesForHMMLearnFolderSuffix')

    if returnExisting == True:
        sampledPoints = ee.FeatureCollection(assetFolder + assetFolderName + assetNamePrefix + '_' + year)
    else:
        labelTree = tp().read_from_json("labelHierarchy.json")
        labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
        labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
        labelNamesL1 = labelNamesL0AndL1AndL2[1]
        labelCodesL1 = labelCodesL0AndL1AndL2[1]
        labelNamesL2 = labelNamesL0AndL1AndL2[2]
        labelCodesL2 = labelCodesL0AndL1AndL2[2]
        l2labelsDict = ee.Dictionary(dict(zip(labelNamesL2, labelCodesL2)))
        # print(l2labelsDict.getInfo(), 'l2labelsDict')
        
        nononeSubset = labelTree.get_labelInfo_atNode(labelNamesL1[0])[1]
        nononeL2labelsL1NumDict = dict(zip(nononeSubset, labelCodesL1[0:1] * len(nononeSubset)))
        oneSubset = labelTree.get_labelInfo_atNode(labelNamesL1[1])[1]
        onel2labelsL1NumDict = dict(zip(oneSubset, labelCodesL1[1:2] * len(oneSubset)))
        l2labelsL1Dict = ee.Dictionary({**nononeL2labelsL1NumDict, **onel2labelsL1NumDict})
        # print(l2labelsL1Dict.getInfo(), 'l2labelsL1Dict')
        
        print('origtrpts path', configCore.get('lulcLabeledPoints'))
        proj = ee.Projection("EPSG:4326").scale(5, 5)
        grids = origTrainingPoints.geometry().bounds().coveringGrid(proj).filterBounds(oneStatesBoundary.geometry())
        for year in yearsToPredict:
            sampledPoints = sampleFeaturesGridwise(grids, year)
            print('exporting ... ', assetNamePrefix + '_' + year)
            if startFreshExport == True:
                if hmmLearnOrClassify == "forHMMLearn":
                    ee.batch.Export.table.toAsset(** {
                        'collection': sampledPoints,
                        'description': assetNamePrefix + '_' + year,
                        'assetId': assetFolder + assetFolderName + hmmLearnFolderSuffix + '/' + assetNamePrefix + '_' + year
                    }).start()
                else:
                    ee.batch.Export.table.toAsset(** {
                        'collection': sampledPoints,
                        'description': assetNamePrefix + '_' + year,
                        'assetId': assetFolder + assetFolderName + '/' + assetNamePrefix + '_' + year
                    }).start()
        
        if hmmLearnOrClassify == "forHMMLearn":
            ptsAsRaster = ee.Image().byte().paint(origTrainingPoints, 1)
            reg = oneStatesBoundary
            ee.batch.Export.image.toAsset(** {
                'image': ptsAsRaster,
                'description': assetNamePrefix + hmmLearnFolderSuffix,
                'assetId': assetFolder + assetFolderName + hmmLearnFolderSuffix + '/' + assetNamePrefix + hmmLearnFolderSuffix,
                'pyramidingPolicy': {".default": "sample"},
                'region': reg.geometry(),
                'scale': scaleOfExport,
                'maxPixels': 1e13}).start()
            print('exporting image mask for HMMLearn', assetNamePrefix + hmmLearnFolderSuffix)
        
    return sampledPoints

