import ee
import math
import configparser as cp
import io
import ast

from features import biophysical
from features import ecological
from utils.trees import TreeParser as tp

def tilewiseRasterExport(imToExport, regionGeom, expOpts):
    # CREATE A GRID ------------------
    # Choose the export CRS
    crs = 'EPSG:24343'
    # Choose the pixel size for export (meters)
    pixelSize = expOpts.get("scale")
    
    # Choose the export tile size (pixels)
    tileSize = 50000
    # Calculate the grid size (meters)
    gridSize = tileSize * pixelSize
    
    # Create the grid covering the geometry bounds
    bounds = regionGeom.bounds(**{'proj': crs, 'maxError': 1})
    grid = bounds.coveringGrid(**{'proj':crs, 'scale': gridSize})
    
    # m.addLayer(grid, {'color': 'blue'}, 'Grid')
    # m

    # CALCULATE CRS TRANSFORM --------
    # Calculate the coordinates of the top-left corner of the grid
    bounds = grid.geometry().bounds(**{'proj': crs, 'maxError': 1})
    
    # Extract the coordinates of the grid
    coordList = ee.Array.cat(bounds.coordinates(), 1)
    
    xCoords = coordList.slice(1, 0, 1)
    yCoords = coordList.slice(1, 1, 2)
    
    # We need the coordinates of the top-left pixel
    xMin = xCoords.reduce('min', [0]).get([0,0])
    yMax = yCoords.reduce('max', [0]).get([0,0])
    
    # Create the CRS Transform
    
    # The transform consists of 6 parameters:
    # [xScale, xShearing, xTranslation, 
    #  yShearing, yScale, yTranslation]
    transform = ee.List([pixelSize, 0, xMin, 0, -pixelSize, yMax]).getInfo()
    # print(transform)

    # Set NoData value
    noDataValue = 0
    exportImage = imToExport.unmask(**{
        'value':noDataValue,
        'sameFootprint': False
    })

    # EXPORT TILES -------------------
    filtered_grid = grid.filter(ee.Filter.intersects('.geo', regionGeom))
    # m.addLayer(
    #     filtered_grid, {'color': 'red'}, 'Filtered Grid')
    # m
    tile_ids = filtered_grid.aggregate_array('system:index').getInfo()
    print('Total tiles', len(tile_ids))
    
    # Export each tile
    descrBase = expOpts.get('descr')
    assetIdBase = expOpts.get('assetPath')
    scl = expOpts.get('scale')
    pyrPol = expOpts.get('pyrPolicy')
    for i, tile_id in enumerate(tile_ids):
        feature = ee.Feature(filtered_grid.toList(1, i).get(0))
        geometry = feature.geometry().intersection(regionGeom, 5)
        task_name = 'tile_' + tile_id.replace(',', '_')
        task = ee.batch.Export.image.toAsset(**{
            'image': exportImage,
            'description': f'{descrBase}_{task_name}',
            'assetId': f'{assetIdBase}_{task_name}',
            'scale': scl,
            'crs': crs,
            'crsTransform': transform,
            'region': geometry,
            'maxPixels': 1e10,
            'pyramidingPolicy': pyrPol})
        task.start()
        print('Started Task: ', i+1)

    return

def predictLabels(yearStr, trainedModel, propNameWithNumLabelsToPredictOn, coarseLevelLabelName, allLabelnamesAndTheirNumLabels, tableWithFeatures, classifierOptions, featuresOptions, resultNewFolderName, resultNewCollName, hmmLearnOrClassify = None):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    #
    configCore = config["CORE"]
    assetFolder     = configCore.get('assetFolderWithFeatures')
    scaleOfExport = configCore.getint('scaleOfMap')
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    existingResultCollName = configClassify.get('existingPredictionsCollection')
    probScaling = configClassify.getint('probabilityScalingFactor')
    probScaling
    configFeaturesAssemble = config["FEATURES-ASSEMBLE"]
    trPtsFolder = configFeaturesAssemble.get('existingLabeledPointsWithFeaturesFolder')
    trPtsMaskImageSuffix = configFeaturesAssemble.get('pointsWithFeaturesForHMMLearnFolderSuffix')
    trPtsTableName = configFeaturesAssemble.get('existingLabeledPointsWithFeatures')
    trPtsMaskImage = ee.Image(assetFolder + trPtsFolder + trPtsMaskImageSuffix + '/' + trPtsTableName + trPtsMaskImageSuffix)
    
    featuresComposite = ecological.assembleAllExistingFeatureRasters(yearStr) \
        .select(featuresOptions.get("names"))
    if hmmLearnOrClassify == "forHMMLearn":
        featuresComposite = featuresComposite.updateMask(trPtsMaskImage)
        print("mask im path: ", assetFolder + trPtsFolder + trPtsMaskImageSuffix + '/' + trPtsTableName + trPtsMaskImageSuffix)
    else:
        featuresComposite = featuresComposite
    
    # Predict over the full feature raster to produce prediction probabilities.
    featureRasterPredicted = ee.Image(featuresComposite.classify(trainedModel, "classProbabilities"))
    
    # Convert the array pixel image into multiband image, where band names correspond to the label names
    # First, sort NameList by ValList; NameList and ValList must be arranged such that
    # name-val correspondence is correct
    labelnamesInFilteredRun = allLabelnamesAndTheirNumLabels.keys()
    labelcodesInFilteredRun = allLabelnamesAndTheirNumLabels.values()
    # The regex in replace() is "JS version" - https://bobbyhadz.com/blog/javascript-remove-special-characters-from-string
    # The "python version" regex worked in jupyter lab console but not in code editor "[\W\_]"
    # Using the JS version since it has to run on server.
    labelnamesInFilteredRunWithProbSuffix = labelnamesInFilteredRun.map(lambda n: ee.String("prob_").cat(ee.String(n).replace("[^a-zA-Z0-9_]", "", "gi")))
    labelsSortedByValues = labelnamesInFilteredRunWithProbSuffix.sort(labelcodesInFilteredRun)
    # print("in nested 2", labelnamesInFilteredRunWithProbSuffix.getInfo())
    # print("in nested 3", labelsSortedByValues.getInfo())

    predictedProbsMultiBand = featureRasterPredicted.arrayFlatten([labelsSortedByValues]).multiply(probScaling).round().uint16()

    # Find top-1 label: label with the maximum probability (decision rule)
    # Sample at labeled points to get predicted labels, for train and test set.
    top1Label = featureRasterPredicted.arrayArgmax().arrayFlatten([["top1LabelIndex"]])
    top1LabelNum = top1Label.remap(ee.List.sequence(0, labelnamesInFilteredRun.size().subtract(1)), labelcodesInFilteredRun.sort()) \
        .rename("top1LabelNum") \
        .uint8()

    featureRasterPredictions = predictedProbsMultiBand.addBands(top1LabelNum).set(dict( \
        year = yearStr, \
        nodeName = coarseLevelLabelName, \
        classifierSchemaFeaturesUsed = trainedModel.schema().join(","), \
        classifierUsed = classifierOptions.get("classifier"), \
        numClassifierTrees = classifierOptions.get("numTrees"), \
        numMaxNodes = classifierOptions.get("maxNodes"), \
        zoneEncoding = featuresOptions.get("zoneEncodingMode"), \
        fractionOfPointsForTrain = classifierOptions.get("trainFraction")))

    if resultNewCollName == None:
        assetCollName = existingResultCollName
    else:
        assetCollName = resultNewCollName

    print("exporting ...", yearStr + " " + coarseLevelLabelName)

    reg = oneStatesBoundary
    # ee.batch.Export.image.toAsset(** {
    #     'image': featureRasterPredictions,
    #     'description': "prediction_" + coarseLevelLabelName + "_" + yearStr,
    #     'assetId': assetFolder + resultNewFolderName + "/" + assetCollName + "/" + "prediction_" + coarseLevelLabelName + "_" + yearStr,
    #     'scale': scaleOfExport,
    #     'region': reg.geometry(),
    #     'maxPixels': 1e12,
    #     "pyramidingPolicy": {".default": "sample"}}).start()
    exportOpts = dict(
        descr = "prediction_" + coarseLevelLabelName + "_" + yearStr,
        assetPath = assetFolder + resultNewFolderName + "/" + assetCollName + "/" + "prediction_" + coarseLevelLabelName + "_" + yearStr,
        scale = scaleOfExport,
        mPix = 1e12,
        pyrPolicy = {".default": "sample"})
    tilewiseRasterExport(featureRasterPredictions, reg.geometry(), exportOpts)

    return featureRasterPredictions

def preparePointsAndLabelsForClassification(allPoints, labelTree, coarseLevelNum = 0, coarseLevelLabelName = "Landcover", fineLevelCodeColName = "labelL1Num", hierarchyMode = "nested", flatHierarchyLevelNum = None):
    fineLevelNum = coarseLevelNum + 1
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    fineLevelNames = labelNamesL0AndL1AndL2[fineLevelNum]
    fineLevelCodes = labelCodesL0AndL1AndL2[fineLevelNum]

    childNames = labelTree.get_labelInfo_atNode(coarseLevelLabelName, mode = "name")[1]
    childCodes = labelTree.get_labelInfo_atNode(coarseLevelLabelName, mode = "code")[1]

    # In case of flat hierarchy, pass through all points, no ramapping. 
    # Get full list of labels at the req level, ignoring nesting.
    if hierarchyMode == "flat":
        childNamesFlat = labelTree.get_labelInfo_byLevel(mode = "name")[flatHierarchyLevelNum]
        childCodesFlat = labelTree.get_labelInfo_byLevel(mode = "code")[flatHierarchyLevelNum]

        pointsForFineLevelHeirarchClassif = allPoints.randomColumn()
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(childNamesFlat, childCodesFlat)
    # In case of level 0, pass through points and labels. No remapping.
    elif hierarchyMode == "nested" and coarseLevelNum == 0:
        # Random column already present in the saved labeled points asset, so not adding here
        pointsForFineLevelHeirarchClassif = allPoints
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(childNames, childCodes)
    # Else, for given coarselevel label,
    # remap non-child (finer level) labels to other
    elif hierarchyMode == "nested":
        otherName = "other"
        otherCode = 99

        fineLevelCodesRemapped = [c if (c in childCodes) else otherCode for c in fineLevelCodes]
        fineLevelNamesRemapped = [c if (c in childNames) else otherName for c in fineLevelNames]
        # Random column already present in the saved labeled points asset, so not adding here
        pointsForFineLevelHeirarchClassif = allPoints \
            .remap(fineLevelCodes, fineLevelCodesRemapped, fineLevelCodeColName)

        fineLevelNamesExt = childNames + [otherName]
        fineLevelCodesExt = childCodes + [otherCode]
        fineLevelNamesAndTheirCodes = ee.Dictionary.fromLists(fineLevelNamesExt, fineLevelCodesExt)

    return pointsForFineLevelHeirarchClassif, fineLevelNamesAndTheirCodes

def trainAndPredictHierarchical_master(hierarchyProcessingOptions = None, featuresOptions = None, classifierOptions = None, roiForClassification = None, resultNewFolderName = None, resultNewCollName = None, returnExisting = False, startFreshExport = False, hmmLearnOrClassify = None):
    def selectClassifier(opts):
        defaultMaxNodes = 600
        # Default GradientBoostedTrees
        if opts.get("classifier") == "RandomForest":
            c = ee.Classifier.smileRandomForest(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": defaultMaxNodes})
        elif opts.get("classifier") == "GradientBoostedTrees":
            c = ee.Classifier.smileGradientTreeBoost(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": opts.get("maxNodes")})
        else:
            c = ee.Classifier.smileGradientTreeBoost(** {"numberOfTrees": opts.get("numTrees"), "maxNodes": opts.get("maxNodes")})
        return c

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    yearsToPredict = ast.literal_eval(configCore.get("annualTimeseriesYears"))
    assetFolder     = configCore.get('assetFolderWithFeatures')
    configFeaturesAssemble = config["FEATURES-ASSEMBLE"]
    assetNamePrefix = configFeaturesAssemble.get('existingLabeledPointsWithFeaturesPrefix')
    assetFolderName = configFeaturesAssemble.get('existingLabeledPointsWithFeaturesFolder')
    trPtsTempDynSuffix = configFeaturesAssemble.get('pointsWithFeaturesForHMMLearnFolderSuffix')

    lulcLevel1CategoriesColName = hierarchyProcessingOptions.get("coarserLevelLabelColumn")
    lulcPalsarHarmnCategoriesColName = hierarchyProcessingOptions.get("finerLevelLabelColumn")

    labelTree = tp().read_from_json("labelHierarchy.json")
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelNamesL1 = labelNamesL0AndL1AndL2[1]
    labelNamesL2 = labelNamesL0AndL1AndL2[2]
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    labelCodesL1 = labelCodesL0AndL1AndL2[1]
    labelCodesL2 = labelCodesL0AndL1AndL2[2]

    labelsListCoarse = labelCodesL1
    propNameWithLabelsCoarse = lulcLevel1CategoriesColName
    propNameWithNumLabelsToPredictOnCoarse = propNameWithLabelsCoarse + "Num"

    propNameWithLabelsFine = lulcPalsarHarmnCategoriesColName
    propNameWithNumLabelsToPredictOnFine = propNameWithLabelsFine + "Num"
    
    # Run classification, a parent node at a time
    trainPtsSplitFilter = ee.Filter.lte('random', classifierOptions.get("trainFraction"))
    for year in yearsToPredict:
        print("====== " + year + " ======")
        if hmmLearnOrClassify == "forHMMLearn":
            pointsWithAllFeaturesYr = ee.FeatureCollection(assetFolder + assetFolderName + trPtsTempDynSuffix + '/' + assetNamePrefix + '_' + year)
        else:
            pointsWithAllFeaturesYr = ee.FeatureCollection(assetFolder + assetFolderName + '/' + assetNamePrefix + '_' + year)
            print(assetFolder + assetFolderName + '/' + assetNamePrefix + '_' + year, 'points path')
        # Level 0
        ptsForL0 = pointsWithAllFeaturesYr.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnCoarse] + ["random"]) \
            .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnCoarse]))
        ptsWithLevel1LabelsPrepped, level1NamesAndCodes = preparePointsAndLabelsForClassification(ptsForL0, labelTree)
        print("l0 names & codes:", level1NamesAndCodes.getInfo())
        # print("l0 hist:", ptsWithLevel1LabelsPrepped.aggregate_histogram(propNameWithNumLabelsToPredictOnCoarse).getInfo())

        classifier = selectClassifier(classifierOptions)
        yearTrainingFraction = ptsWithLevel1LabelsPrepped.filter(trainPtsSplitFilter)
        print(pointsWithAllFeaturesYr.getString('system:id').getInfo(), ' year points asset folder')
        # print(ptsWithLevel1LabelsPrepped.first().propertyNames().getInfo(), 'ptsWithLevel1LabelsPrepped prop names')
        # print(yearTrainingFraction.size().getInfo(), year + ' trainingFraction size')
        print(featuresOptions.get("names"), 'selected features')
        yearTrainedModel = classifier.train(** { \
            'features': yearTrainingFraction, \
            'classProperty': propNameWithNumLabelsToPredictOnCoarse, \
            'inputProperties': featuresOptions.get("names"),
        }).setOutputMode('MULTIPROBABILITY')
        if hmmLearnOrClassify == "forHMMLearn":
            predictLabels(year, yearTrainedModel, propNameWithNumLabelsToPredictOnCoarse, 'l0', level1NamesAndCodes, ptsWithLevel1LabelsPrepped, classifierOptions, featuresOptions, resultNewFolderName, resultNewCollName, hmmLearnOrClassify = "forHMMLearn")
        else:
            predictLabels(year, yearTrainedModel, propNameWithNumLabelsToPredictOnCoarse, 'l0', level1NamesAndCodes, ptsWithLevel1LabelsPrepped, classifierOptions, featuresOptions, resultNewFolderName, resultNewCollName)
        
        # All in level 1, in a loop
        for labL1 in labelNamesL1:
            ptsForL1 = pointsWithAllFeaturesYr.select(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine] + ["random"]) \
                .filter(ee.Filter.notNull(featuresOptions.get("names") + [propNameWithNumLabelsToPredictOnFine]))
            ptsWithFineLevelLabelsPrepped, fineLevelNamesAndCodesPrepped = preparePointsAndLabelsForClassification(ptsForL1, labelTree, 1, labL1, propNameWithNumLabelsToPredictOnFine)
            print("l1 names & codes:", fineLevelNamesAndCodesPrepped.getInfo())
            # print("l1 hist:", ptsWithFineLevelLabelsPrepped.aggregate_histogram(propNameWithNumLabelsToPredictOnFine).getInfo())
            yearTrainingFractionFine = ptsWithFineLevelLabelsPrepped.filter(trainPtsSplitFilter)
            print(ptsWithFineLevelLabelsPrepped.first().propertyNames().getInfo(), 'ptsWithFineLevelLabelsPrepped prop names')
            # print(yearTrainingFractionFine.size().getInfo(), year + ' trainingFraction size')
            yearTrainedModelFine = classifier.train(** { \
                'features': yearTrainingFractionFine, \
                'classProperty': propNameWithNumLabelsToPredictOnFine, \
                'inputProperties': featuresOptions.get("names"),
            }).setOutputMode('MULTIPROBABILITY')
            
            if hmmLearnOrClassify == "forHMMLearn":
                predictLabels(year, yearTrainedModelFine, propNameWithNumLabelsToPredictOnFine, labL1, fineLevelNamesAndCodesPrepped, ptsWithFineLevelLabelsPrepped, classifierOptions, featuresOptions, resultNewFolderName, resultNewCollName, hmmLearnOrClassify = "forHMMLearn")
            else:
                predictLabels(year, yearTrainedModelFine, propNameWithNumLabelsToPredictOnFine, labL1, fineLevelNamesAndCodesPrepped, ptsWithFineLevelLabelsPrepped, classifierOptions, featuresOptions, resultNewFolderName, resultNewCollName)

    print("hierarchy mode explicit")

    return ptsForL0.size()

def yearwiseHierarchicalPrediction(yearStr, expHierL1ProbsScaled, expHierL2NononeProbsScaled, expHierL2OneProbsScaled, resultNewFolderName, resultNewCollName):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    configCore = config["CORE"]
    assetFolder     = configCore.get('assetFolderWithFeatures')
    scaleOfExport = configCore.getint('scaleOfMap')
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    probScaling = configClassify.getint('probabilityScalingFactor')
    
    labelTree = tp().read_from_json("labelHierarchy.json")
    coarseLevelNum = 1
    fineLevelNum = coarseLevelNum + 1
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    coarseLevelNames = labelNamesL0AndL1AndL2[coarseLevelNum]
    coarseLevelCodes = labelCodesL0AndL1AndL2[coarseLevelNum]
    fineLevelNames = labelNamesL0AndL1AndL2[fineLevelNum]
    fineLevelCodes = labelCodesL0AndL1AndL2[fineLevelNum]

    oneGroupNums = labelTree.get_labelInfo_atNode("one", mode = "code")[1]
    nononeGroupNums = labelTree.get_labelInfo_atNode("nonone", mode = "code")[1]
    oneNum = labelTree.find_by_name("one").code
    nononeNum = labelTree.find_by_name("nonone").code
    
    # List of class probabilitiy band names, in prep for combining probabilities
    probBandNamesL1 = expHierL1ProbsScaled.bandNames().removeAll(['top1LabelNum'])
    probBandNamesL2One    = expHierL2OneProbsScaled.bandNames().removeAll(['top1LabelNum', 'prob_other'])
    probBandNamesL2Nonone = expHierL2NononeProbsScaled.bandNames().removeAll(['top1LabelNum', 'prob_other'])

    # Remove "prob_" prefix
    l1ClassNames = probBandNamesL1.map(lambda n: ee.String(n).slice(5))
    l2OneClassNames = probBandNamesL2One.map(lambda n: ee.String(n).slice(5))
    l2NononeClassNames = probBandNamesL2Nonone.map(lambda n: ee.String(n).slice(5))

    # Class labels and their codes, according to original convention, for each level
    allClassNamesSorted = l2OneClassNames.cat(l2NononeClassNames).sort()
    l1ClassNamesSorted = l1ClassNames.sort()
    l2OneClassNamesSorted = l2OneClassNames.sort()
    l2NononeClassNamesSorted = l2NononeClassNames.sort()
    l2OneClassCodesForSortedLabels = l2OneClassNamesSorted.map(lambda n: allClassNamesSorted.indexOf(ee.String(n)).add(1))
    l2NononeClassCodesForSortedLabels = l2NononeClassNamesSorted.map(lambda n: allClassNamesSorted.indexOf(ee.String(n)).add(1))

    ####### Readjust probabilities ######
    expHierL1Probs = expHierL1ProbsScaled.divide(probScaling)
    # print(expHierL1ProbsScaled.getInfo(), 'expHierL1ProbsScaled')
    # print(expHierL1Probs.getInfo(), 'expHierL1Probs')
    expHierL2NononeProbs = expHierL2NononeProbsScaled.divide(probScaling)
    expHierL2OneProbs = expHierL2OneProbsScaled.divide(probScaling)

    l1OneReadjust = expHierL1Probs.select("prob_one") \
        .multiply(ee.Image(1).subtract(expHierL2OneProbs.select("prob_other")))
        # .rename("one_readjust");
    # print(l1OneReadjust.getInfo(), 'l1OneReadjust')
    l1NononeReadjust = expHierL1Probs.select("prob_nonone") \
        .multiply(ee.Image(1).subtract(expHierL2NononeProbs.select("prob_other")))
        # .rename("nonone_readjust");
    # standardise (sum to 1) readjusted l1 probs 
    l1Readjust = ee.Image.cat([l1NononeReadjust, l1OneReadjust]) \
        .divide(l1NononeReadjust.add(l1OneReadjust))
    # print(l1Readjust.getInfo(), 'l1Readjust')

    # standardise (sum to 1) readjusted l2 probs
    l2OneReadjust = expHierL2OneProbs.select(probBandNamesL2One) \
        .divide(expHierL2OneProbs.select(probBandNamesL2One).reduce(ee.Reducer.sum()))
    l2NononeReadjust = expHierL2NononeProbs.select(probBandNamesL2Nonone) \
        .divide(expHierL2NononeProbs.select(probBandNamesL2Nonone).reduce(ee.Reducer.sum()))
    # print(l2OneReadjust.getInfo(), 'l2OneReadjust')

    ###### Hierarchical multiplicative rule
    # Multiply L1 probabilities with their corresponding set of probabilities from L2
    l1MultL2One = l1Readjust.select('prob_one') \
        .multiply(l2OneReadjust.select(probBandNamesL2One))\
        .rename(l2OneClassNames)
    l1MultL2Nonone = l1Readjust.select('prob_nonone') \
        .multiply(l2NononeReadjust.select(probBandNamesL2Nonone)) \
        .rename(l2NononeClassNames)

    # Merge all these probabilities and calculate their top 1 label
    allProbsInArray = l1MultL2One.addBands(l1MultL2Nonone) \
        .select(allClassNamesSorted) \
        .toArray()
    expHierMultL2 = allProbsInArray.arrayArgmax().arrayFlatten([["top1LabelIndexHei"]]).uint8() \
        .remap(ee.List.sequence(0, len(fineLevelNames)-1), ee.List.sequence(1, len(fineLevelNames)))
    expHierMultL1 = expHierMultL2.remap(** {
        "from": oneGroupNums + nononeGroupNums,
        "to": [oneNum]*len(oneGroupNums) + [nononeNum]*len(nononeGroupNums), 
        "bandName": 'remapped'})
        
    expHierMultAllLevelLabels = ee.Image.cat([expHierMultL1.rename("l1LabelNum"), expHierMultL2.rename("l2LabelNum")])

    # Find prob of top1 label
    expHierMultL2Confidence = allProbsInArray.arrayReduce(ee.Reducer.max(), [0]) \
        .arrayFlatten([["probL2Label"]])
    expHierMultL1NononeProb = ee.Image(0).where(expHierMultL1.eq(nononeNum), l1MultL2Nonone.reduce(ee.Reducer.sum()))
    expHierMultL1OneProb    = ee.Image(0).where(expHierMultL1.eq(oneNum), l1MultL2One.reduce(ee.Reducer.sum()))
    expHierMultL1Confidence = expHierMultL1NononeProb.add(expHierMultL1OneProb).rename("probL1Label")
    
    expHierMultAllLevelTop1Conf = ee.Image.cat([expHierMultL1Confidence, expHierMultL2Confidence])

    # For all levels, combine top 1 labels with their probs, and probs of all labels
    outputSuffix = "expHierMult"
    probBandNamesL2OneWithL1Prefix    =    probBandNamesL2One.map(lambda n: ee.String('prob_one_').cat(ee.String(n).slice(5)))
    probBandNamesL2NononeWithL1Prefix = probBandNamesL2Nonone.map(lambda n: ee.String('prob_nonone_').cat(ee.String(n).slice(5)))
    expHierMultAllLevelLabelsAllConfsProbs = ee.Image.cat([ \
        expHierMultL2.rename("l2LabelNum").uint8(), \
        l1MultL2One.select(l2OneClassNames, probBandNamesL2OneWithL1Prefix).multiply(probScaling).round().uint16(), \
        l1MultL2Nonone.select(l2NononeClassNames, probBandNamesL2NononeWithL1Prefix).multiply(probScaling).round().uint16()]) \
        .set(dict(year = yearStr, nodeName = outputSuffix))

    print('exporting ' + outputSuffix + ' for year ' + yearStr + '...')
    
    reg = oneStatesBoundary
    # ee.batch.Export.image.toAsset(** {
    #     'image': expHierMultAllLevelLabelsAllConfsProbs,
    #     'description': "prediction_" + outputSuffix + "_" + yearStr,
    #     'assetId': assetFolder + resultNewFolderName + "/" + resultNewCollName + "/" + "prediction_" + outputSuffix + "_" + yearStr,
    #     'scale': scaleOfExport,
    #     'region': reg.geometry(),
    #     'maxPixels': 1e12,
    #     "pyramidingPolicy": {".default": "sample"}}).start()
    exportOpts = dict(
        descr = "prediction_" + outputSuffix + "_" + yearStr,
        assetPath = assetFolder + resultNewFolderName + "/" + resultNewCollName + "/" + "prediction_" + outputSuffix + "_" + yearStr,
        scale = scaleOfExport,
        mPix = 1e12,
        pyrPolicy = {".default": "sample"})
    tilewiseRasterExport(expHierMultAllLevelLabelsAllConfsProbs, reg.geometry(), exportOpts)

    return expHierMultAllLevelLabelsAllConfsProbs

def annualHierarchicalPredictions(resultNewFolderName, resultNewCollName):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    
    configCore = config["CORE"]
    yearsToPredict = ast.literal_eval(configCore.get("annualTimeseriesYears"))
    assetFolder = configCore.get('assetFolderWithFeatures')
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    existingResultCollName = configClassify.get('existingPredictionsCollection')

    if resultNewCollName == None:
        assetCollName = existingResultCollName
    else:
        assetCollName = resultNewCollName

    # print('icoll ', assetFolder + resultNewFolderName + "/" + assetCollName)
    predHeirProbsColl = ee.ImageCollection(assetFolder + resultNewFolderName + "/" + assetCollName)
    print(predHeirProbsColl.getString('system:id').getInfo(), 'predHeirProbsColl id')

    for year in yearsToPredict:
        yearExpHierL1Probs = predHeirProbsColl.filter(ee.Filter.And(ee.Filter.eq('year', year), ee.Filter.eq('nodeName', 'l0'))).mosaic()
        yearExpHierL2NononeProbs = predHeirProbsColl.filter(ee.Filter.And(ee.Filter.eq('year', year), ee.Filter.eq('nodeName', 'nonone'))).mosaic()
        yearExpHierL2OneProbs = predHeirProbsColl.filter(ee.Filter.And(ee.Filter.eq('year', year), ee.Filter.eq('nodeName', 'one'))).mosaic()
        yearwiseHierarchicalPrediction(year, yearExpHierL1Probs, yearExpHierL2NononeProbs, yearExpHierL2OneProbs, resultNewFolderName, resultNewCollName)

    return predHeirProbsColl

def optimalTimeseriesLabels(resultFolderName = None):
    # Taken from JS script https://code.earthengine.google.com/e6be9db93645e0f34e6b2bb4fa036779?noload=1
    def sampArrImg(arrImg):
        # return ee.Image(arrImg).sample(dummyPt, 10).first()
        return ee.Image(arrImg).sample(dummyPt, 10).first().get('constant')
        
    def sampArrImg2(arrImg):
        return ee.Image(arrImg).sample(dummyPt, 10).first().get('array')
    
    def removeProbTags(s):
        rep1 = ee.String(s).replace('prob_one_', '')
        rep2 = rep1.replace('prob_nonone_', '')
        return rep2
    
    def viterbiDecoder(probArrIm, stepsPosteriorProb):
        probsArr = ee.Image(stepsPosteriorProb)
        # print(sampArrImg(probsArr).getInfo(), 'probsArr')
        
        # Take this step's observation probe ... 
        nextObsProbs = ee.Image(probArrIm)
        # print('numClasses', numClasses)
        nextObsProbs_reshaped = nextObsProbs.arrayRepeat(1, numClasses) \
            .arrayReshape(ee.Image([numClasses*numClasses, 1]).toArray(), 2)
        # print(sampArrImg2(nextObsProbs_reshaped).getInfo(), 'nextObsProbs_reshaped')
        
        # Pick up posterior probs of previous step
        prevPosteriorProb = probsArr.arraySlice(1, -1)
        # print(sampArrImg(prevPosteriorProb), 'prevPosteriorProb')
        # Reshape them for aligned mult with observation probs of this step and transition probs
        prevPosteriorProbs_reshaped = prevPosteriorProb.arrayRepeat(0, numClasses)
        # print(sampArrImg(prevPosteriorProbs_reshaped), 'prevPosteriorProbs_reshaped')

        # Aligned multiply (add logs of) prev posteior, current observation and transition probs 
        nextPosteriorProbs_long = prevPosteriorProbs_reshaped \
            .add(transitionProbs_reshaped) \
            .add(nextObsProbs_reshaped)
        # print(sampArrImg(nextPosteriorProbs_long), 'nextPosteriorProbs_long')
        
        # Reshape the result into 2x2 to get probs of each class in a separate row
        nextPosteriorProbs_wide = nextPosteriorProbs_long.arrayReshape(ee.Image([numClasses, numClasses]).toArray(), 2)
        # print(sampArrImg(nextPosteriorProbs_wide), 'nextPosteriorProbs_wide')
        # For each row (class) find the max prob
        nextPosteriorProbs_best = nextPosteriorProbs_wide.arrayReduce(ee.Reducer.max(), [1])
        
        probsArr = probsArr.toFloat().arrayCat(nextPosteriorProbs_best.toFloat(), 1)
        
        return probsArr

    def removeProbTags(s):
        sSplits = ee.String(s).split('_')
        return sSplits.slice(2).join("_")

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))
    
    configCore = config["CORE"]
    yearsToPredict = ast.literal_eval(configCore.get("annualTimeseriesYears"))[1:]
    yearsToPredict.sort()
    assetFolder = configCore.get('assetFolderWithFeatures')
    oneStatesBoundary = ee.Feature(ee.FeatureCollection(configCore.get('oneStates')).first())
    scaleOfExport = configCore.getint('scaleOfMap')
    configZonesAgroecol = config["AOI-CLASSIFICATION-ZONES-AGROECOLOGICAL-ZONES"]
    zonationAssetName = configZonesAgroecol.get("existingZonesNumeric")
    configClassify = config["CLASSIFICATION-TRAIN&PREDICT"]
    existingResultCollName = configClassify.get('existingPredictionsCollection')
    probScaling = configClassify.getint('probabilityScalingFactor')
    timeseriesLabelsIm = configClassify.get('existingPredictionsOptimalLabelSequence')
    
    agroEcolZonationIm = ee.Image(assetFolder + zonationAssetName)

    hmmParamsFileName = configClassify.get('hmmParametersFileName')
    with open(hmmParamsFileName + ".ini", 'r') as f2:
        fileContents2 = f2.read()
    hmmParamsObj = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    hmmParamsObj.read_file(io.StringIO(fileContents2))

    paramsAer2 = hmmParamsObj["AER2"]
    hmmTransitionsAer2 = ast.literal_eval(paramsAer2.get("aer2_hmmTransitionProbabilities"))
    paramsAer3 = hmmParamsObj["AER3"]
    hmmTransitionsAer3 = ast.literal_eval(paramsAer3.get("aer3_hmmTransitionProbabilities"))
    paramsAer4 = hmmParamsObj["AER4"]
    hmmTransitionsAer4 = ast.literal_eval(paramsAer4.get("aer4_hmmTransitionProbabilities"))
    paramsAer5 = hmmParamsObj["AER5"]
    hmmTransitionsAer5 = ast.literal_eval(paramsAer5.get("aer5_hmmTransitionProbabilities"))
    paramsAer6 = hmmParamsObj["AER6"]
    hmmTransitionsAer6 = ast.literal_eval(paramsAer6.get("aer6_hmmTransitionProbabilities"))
    paramsAer7 = hmmParamsObj["AER7"]
    hmmTransitionsAer7 = ast.literal_eval(paramsAer7.get("aer7_hmmTransitionProbabilities"))
    paramsAer8 = hmmParamsObj["AER8"]
    hmmTransitionsAer8 = ast.literal_eval(paramsAer8.get("aer8_hmmTransitionProbabilities"))
    paramsAer9 = hmmParamsObj["AER9"]
    hmmTransitionsAer9 = ast.literal_eval(paramsAer9.get("aer9_hmmTransitionProbabilities"))
    paramsAer10 = hmmParamsObj["AER10"]
    hmmTransitionsAer10 = ast.literal_eval(paramsAer10.get("aer10_hmmTransitionProbabilities"))
    paramsAer11 = hmmParamsObj["AER11"]
    hmmTransitionsAer11 = ast.literal_eval(paramsAer11.get("aer11_hmmTransitionProbabilities"))
    paramsAer12 = hmmParamsObj["AER12"]
    hmmTransitionsAer12 = ast.literal_eval(paramsAer12.get("aer12_hmmTransitionProbabilities"))
    paramsAer13 = hmmParamsObj["AER13"]
    hmmTransitionsAer13 = ast.literal_eval(paramsAer13.get("aer13_hmmTransitionProbabilities"))
    paramsAer14 = hmmParamsObj["AER14"]
    hmmTransitionsAer14 = ast.literal_eval(paramsAer14.get("aer14_hmmTransitionProbabilities"))
    paramsAer18 = hmmParamsObj["AER18"]
    hmmTransitionsAer18 = ast.literal_eval(paramsAer18.get("aer18_hmmTransitionProbabilities"))
    paramsAer19 = hmmParamsObj["AER19"]
    hmmTransitionsAer19 = ast.literal_eval(paramsAer19.get("aer19_hmmTransitionProbabilities"))
    
    transIm = agroEcolZonationIm.eq(2).multiply(ee.Image(ee.Array(hmmTransitionsAer2))) \
        .add(agroEcolZonationIm.eq(3).multiply(ee.Image(ee.Array(hmmTransitionsAer3)))) \
        .add(agroEcolZonationIm.eq(4).multiply(ee.Image(ee.Array(hmmTransitionsAer4)))) \
        .add(agroEcolZonationIm.eq(5).multiply(ee.Image(ee.Array(hmmTransitionsAer5)))) \
        .add(agroEcolZonationIm.eq(6).multiply(ee.Image(ee.Array(hmmTransitionsAer6)))) \
        .add(agroEcolZonationIm.eq(7).multiply(ee.Image(ee.Array(hmmTransitionsAer7)))) \
        .add(agroEcolZonationIm.eq(8).multiply(ee.Image(ee.Array(hmmTransitionsAer8)))) \
        .add(agroEcolZonationIm.eq(9).multiply(ee.Image(ee.Array(hmmTransitionsAer9)))) \
        .add(agroEcolZonationIm.eq(10).multiply(ee.Image(ee.Array(hmmTransitionsAer10)))) \
        .add(agroEcolZonationIm.eq(11).multiply(ee.Image(ee.Array(hmmTransitionsAer11)))) \
        .add(agroEcolZonationIm.eq(12).multiply(ee.Image(ee.Array(hmmTransitionsAer12)))) \
        .add(agroEcolZonationIm.eq(13).multiply(ee.Image(ee.Array(hmmTransitionsAer13)))) \
        .add(agroEcolZonationIm.eq(14).multiply(ee.Image(ee.Array(hmmTransitionsAer14)))) \
        .add(agroEcolZonationIm.eq(18).multiply(ee.Image(ee.Array(hmmTransitionsAer18)))) \
        .add(agroEcolZonationIm.eq(19).multiply(ee.Image(ee.Array(hmmTransitionsAer19)))) \
        .log10()

    # Consider probabilities from the 2017 map as prior probabilitie.
    # Arrange probability bands in the order of their numeric values.
    priorIm = ee.ImageCollection(assetFolder + resultFolderName + "/" + existingResultCollName) \
        .filter(ee.Filter.eq('nodeName', 'expHierMult')) \
        .filter(ee.Filter.eq('year', '2017')) \
        .select(['prob_nonone_agri_open', 'prob_one_bare_rocky', 
                 'prob_nonone_built', 'prob_nonone_cultivated_trees', 
                 'prob_one_dunes', 'prob_nonone_forest', 
                 'prob_one_saline_flat', 'prob_one_savanna_grass', 
                 'prob_one_savanna_shrub', 'prob_one_savanna_tree', 
                 'prob_nonone_water_wetland']) \
        .mosaic() \
        .toArray(0) \
        .log10()
    predAnnProbsCollWideTiled = ee.ImageCollection(assetFolder + resultFolderName + "/" + existingResultCollName) \
        .filter(ee.Filter.eq('nodeName', 'expHierMult')) \
        .filter(ee.Filter.neq('year', '2017')) \
        .select(['prob_one_.*|prob_nonone_.*'])
    predAnnProbsCollWide = ee.ImageCollection.fromImages( \
        ee.List(yearsToPredict).map(lambda y: predAnnProbsCollWideTiled.filter(ee.Filter.eq('year', y)).mosaic()))
    # Order the prob values according to the sequence of their label codes.
    # First, remove prob tags from band names, then reorder the bands alphabetically (which is same as sequence of the label codes)
    imBandNamesWithProbTag = predAnnProbsCollWideTiled.first().bandNames()
    imBandNamesWithoutProbTag = imBandNamesWithProbTag.map(removeProbTags)
    predAnnProbsColl = predAnnProbsCollWide \
        .map(lambda im: im.divide(probScaling).select(imBandNamesWithProbTag, imBandNamesWithoutProbTag).select(imBandNamesWithoutProbTag.sort()).toArray(0).log10())
    
    numViterbiSteps = len(yearsToPredict)
    
    dummyColumn = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    dummyPt = ee.Geometry.Point([76.06073324693693,13.604930480165638])
    predAnnProbsSortedList = predAnnProbsColl \
        .sort('year') \
        .toList(numViterbiSteps)

    labelTree = tp().read_from_json("labelHierarchy.json")
    coarseLevelNum = 1
    fineLevelNum = coarseLevelNum + 1
    labelNamesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "name")
    labelCodesL0AndL1AndL2 = labelTree.get_labelInfo_byLevel(mode = "code")
    coarseLevelNames = labelNamesL0AndL1AndL2[coarseLevelNum]
    coarseLevelCodes = labelCodesL0AndL1AndL2[coarseLevelNum]
    fineLevelNames = labelNamesL0AndL1AndL2[fineLevelNum]
    fineLevelCodes = labelCodesL0AndL1AndL2[fineLevelNum]
    
    numClasses = len(fineLevelNames)
    # print('numClasses another try', len(fineLevelNames))
    transitionProbs_reshaped = transIm.arrayTranspose().arrayReshape(ee.Image([numClasses*numClasses, 1]).toArray(), 2)    
    
    # STEP 1
    print('------ Step 1 ------')
    # Step 1
    # array to hold probs while stepping through dummy column,
    # initialized to a dummy column
    stepsPosteriorProb = ee.Image(ee.Array(dummyColumn))
    # print(sampArrImg(stepsPosteriorProb).getInfo(), 'log probs thru steps, dummy col init')
    
    # Aligned multiply (add logs of) prior and current observation probs
    start = priorIm.add(ee.Image(predAnnProbsSortedList.get(0)))
    # print('startIm', start.getInfo())
    # print(sampArrImg(start).getInfo(), 'step 1 log probs')
    # print(sampArrImg2(start).getInfo(), '2... step 1 log probs')
    stepsPosteriorProb = stepsPosteriorProb.arrayCat(start, 1)
    # print(sampArrImg(stepsPosteriorProb).getInfo(), 'log probs thru steps, step 1 appended')
    # print(sampArrImg2(stepsPosteriorProb).getInfo(), '2.. log probs thru steps, step 1 appended')
    
    print('------ Steps 2-7 ------')
    # decodedLabels = viterbiDecoder(predAnnProbsSortedList.get(2), stepsPosteriorProb)
    decodedLabels = predAnnProbsSortedList.slice(1).iterate(viterbiDecoder, stepsPosteriorProb)
    # print(sampArrImg(decodedLabels).getInfo(), 'decodedLabels res')

    stepsDone = ee.Image(decodedLabels).arrayLength(1).subtract(1).toInt()
    print(sampArrImg(stepsDone).getInfo(), 'stepsDone')
    probsArr = ee.Image(decodedLabels).arraySlice(1, 1)
    # print(sampArrImg(ee.Image(10).pow(probsArr)).getInfo(), 'probsArr final')

    maxProbsByCol = probsArr.arrayReduce(ee.Reducer.max(), [0])
    # print('max probs by step', sampArrImg(ee.Image(10).pow(maxProbsByCol)).getInfo())
    maxProbsByColRepeated = maxProbsByCol.arrayRepeat(0, numClasses)
    # print('max probs by col rep', sampArrImg(maxProbsByColRepeated))
    maxProbsByColChecked = probsArr.eq(maxProbsByColRepeated)
    # print('max probs by col chk', sampArrImg(maxProbsByColChecked))

    rowPos = ee.Image(ee.Array(list(range(1, 11+1))))
    rowPosRep = rowPos.arrayRepeat(1, numViterbiSteps)
    # print('rowPosRep', sampArrImg(rowPosRep))
    maxProbLabelsByCol = maxProbsByColChecked.multiply(rowPosRep) \
        .arrayReduce(ee.Reducer.max(), [0])
    # print('max prob label seq', sampArrImg(maxProbLabelsByCol).getInfo())
    yearBandNames = ["y" + year for year in yearsToPredict]
    optLabelSeqInBands = maxProbLabelsByCol.arrayProject([1]).arrayFlatten([yearBandNames]).toUint8() \
        .set(dict(year = 'years2to8', nodeName = "optLabelSeq_2017prior"))
    # print('optLabelSeqInBands', optLabelSeqInBands.getInfo())

    reg = oneStatesBoundary
    # ee.batch.Export.image.toAsset(** {
    #     'image': optLabelSeqInBands,
    #     'description': timeseriesLabelsIm + "_allYears",
    #     'assetId': assetFolder + resultFolderName + "/" + existingResultCollName + "/" + timeseriesLabelsIm,
    #     'pyramidingPolicy': {".default": "sample"},
    #     'region': reg.geometry(),
    #     'scale': scaleOfExport,
    #     'maxPixels': 1e12}).start()
    exportOpts = dict(
        descr = timeseriesLabelsIm,
        assetPath = assetFolder + resultFolderName + "/" + existingResultCollName + "/" + timeseriesLabelsIm,
        scale = scaleOfExport,
        mPix = 1e12,
        pyrPolicy = {".default": "sample"})
    print(assetFolder + resultFolderName + "/" + existingResultCollName + "/" + timeseriesLabelsIm, 'result export coll')
    tilewiseRasterExport(optLabelSeqInBands, reg.geometry(), exportOpts)

    return