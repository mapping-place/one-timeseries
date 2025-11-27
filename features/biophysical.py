import ee
import configparser as cp
import ast
import io

def classificationZonesFromRegionsFeatureCollection(fc = None, zoneBandNamePrefix = None):
    zonesNumeric = fc.reduceToImage(["zoneNum"], ee.Reducer.first()) \
        .toUint8() \
        .rename(zoneBandNamePrefix + "Num")

    classifZones = zonesNumeric

    return classifZones

def classificationZonesFromAgroecologicalNumeric(returnExisting = False, startFreshExport = False):
    def assignNumberToAgroecolZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allAgroecolFC.filter(ee.Filter.eq("physio_reg", agroecolNames.get(indexOfiIntoLists))).union().first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    assetFolder = configCore.get('assetFolderWithFeatures')
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    scaleOfExport = configCore.getint('scaleOfMap')
    configZonesAgroecol = config["AOI-CLASSIFICATION-ZONES-AGROECOLOGICAL-ZONES"]
    assetName = configZonesAgroecol.get("existingZonesNumeric")
    allAgroecolFC = ee.FeatureCollection(configZonesAgroecol.get("indiaAgroEcologicalRegions"))
    zoneBandnamePrefix = configZonesAgroecol.get("featureBandNamePrefix")
    agroecolNames = ee.List(ast.literal_eval(configZonesAgroecol.get("AgroecolNames")))

    reg = oneStatesBoundary
    
    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
        # print(assetFolder + assetName)
    else:
        # Assign a numeric and label to each Agroecol feature, rasterize it
        zoneNumberSeq = ee.List.sequence(1, agroecolNames.length())
        AgroecolWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToAgroecolZone))
        AgroecolZonesNum = classificationZonesFromRegionsFeatureCollection(AgroecolWithZoneNumberAssigned, zoneBandnamePrefix)
        zones = AgroecolZonesNum

        if startFreshExport == True:
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': scaleOfExport,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones.clip(reg)
