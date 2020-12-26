import os
import imp
import copy
import torch
import random
import numpy as np

class _layer:
  name = ''
  _type = ''
  birthday = 0
  dimension = []
  dictDimensionInput = {}
  dimensionInputResult = []
  dimensionOutputResult = []

  def __init__(self, name, _type, dimension, birthday = 0, dimensionInputResult = []):
    self.name = name
    self._type = _type
    self.dimension = dimension
    self.birthday = birthday
    if not (_type == 'Input' or _type == 'Output' or _type == 'Conv2d' or _type == 'Linear' or _type == 'ConvTranspose2d'):
      print ('unknown layer type for layer:', name)
      exit()
    # Conv2d dimension: [in_channels, out_channels, kernel_size, stride, padding]
    # Linear dimension: [in_features, out_features]
    # ConvTranspose2d dimension: [in_channels, out_channels, kernel_size, stride, padding]
    if _type == 'ConvTranspose2d':
      if not dimensionInputResult or len(dimensionInputResult) != 3:
        print ('\n\'ConvTranspose2d\' layer (' + name + ') must specify 3D input dimension!')
        exit()
      self.dimensionInputResult = dimensionInputResult
    if _type == 'Input':
      self.dimensionOutputResult = dimension
    elif _type == 'Output':
      self.dimensionInputResult = dimension
    if _type == 'Input' or _type == 'Output':
      if len(dimension) == 1: # only out_features is defined
        self.dimension = [0, dimension[0]]
      elif len(dimension) == 3: # only out_channels is defined
        self.dimension = [0, dimension[2], 0, 0, 0]

class _net:
  name = ''
  listNameInput = []
  listNameOutput = []
  listNameLayer = []
  dictLayer = {}
  dictGraph = {}
  birthdayLatest = 0
  # non-essential
  dictGraphReversed = {}
  # pyTorch related
  modelNow = None
  modelNext = None
  actionOnModel = 'None'
  dictDiffLayer = {}

  def __init__(self, name, listNameInput, listNameOutput, listNameLayer, dictLayer, dictGraph):
    self.name = name
    self.listNameInput = listNameInput
    self.listNameOutput = listNameOutput
    self.listNameLayer = listNameLayer
    self.dictLayer = dictLayer
    self.dictGraph = dictGraph

  def exitGateway(self, number):
    for nameLayer in self.listNameLayer:
      layer = self.dictLayer[nameLayer]
      print ()
      print (nameLayer, layer._type, layer.dimension)
      print (layer.dictDimensionInput)
      print (layer.dimensionInputResult)
      print (layer.dimensionOutputResult)
    print ('\ninput(s):', self.listNameInput)
    print ('\nlayer(s):', self.listNameLayer)
    print ('\noutput(s):', self.listNameOutput)
    print ('\ngraph:\n', self.dictGraph)
    print ('\nreversed graph:\n', self.dictGraphReversed)
    print ('exit gateway:', number)
    exit()

  def hasValidConnection(self, nameLayer):
    typeLayer = self.dictLayer[nameLayer]._type
    for nameLayerForward in self.dictGraph[nameLayer]:
      typeLayerForward = self.dictLayer[nameLayerForward]._type
      if typeLayer == 'Conv2d':
        if not (typeLayerForward == 'Conv2d' or typeLayerForward == 'Linear' or typeLayerForward == 'Output'):
          self.exitGateway(1)
      elif typeLayer == 'Linear':
        if not (typeLayerForward == 'Linear' or typeLayerForward == 'ConvTranspose2d' or typeLayerForward == 'Output'):
          self.exitGateway(2)
      elif typeLayer == 'ConvTranspose2d':
        if not (typeLayerForward == 'ConvTranspose2d' or typeLayerForward == 'Output'):
          self.exitGateway(3)
      elif typeLayer == 'Input':
        if typeLayerForward == 'Output':
          self.exitGateway(4)
      else:
        self.exitGateway(5)
    return True

  def validateConnection(self):
    for nameInput in self.listNameInput:
      self.hasValidConnection(nameInput)
    for nameLayer in self.listNameLayer:
      self.hasValidConnection(nameLayer)
    print ('\n@@@ all connection are valid @@@')
    return True

  def computeReversedGraph(self):
    for nameLayer in self.listNameLayer:
      self.dictGraphReversed[nameLayer] = []
    for nameOutput in self.listNameOutput:
      self.dictGraphReversed[nameOutput] = []
    for nameLayerStart in self.dictGraph:
      for nameLayerEnd in self.dictGraph[nameLayerStart]:
        self.dictGraphReversed[nameLayerEnd].append(nameLayerStart)

  def computeReversedGraphAlone(self, dictGraph):
    dictReversed = {}
    for nameLayer in self.listNameLayer:
      dictReversed[nameLayer] = []
    for nameOutput in self.listNameOutput:
      dictReversed[nameOutput] = []
    for nameLayerStart in dictGraph:
      for nameLayerEnd in dictGraph[nameLayerStart]:
        dictReversed[nameLayerEnd].append(nameLayerStart)
    return dictReversed

  def computeLayerPrecedence(self):
    self.dictGraphReversed = {}
    self.computeReversedGraph()
    listNameLayer = []
    listNameLayerNotReady = copy.deepcopy(self.listNameLayer)
    while listNameLayerNotReady:
      restartLoop = False
      for nameLayer in listNameLayerNotReady:
        isReady = True
        for nameLayerSource in self.dictGraphReversed[nameLayer]:
          if not ('input' in nameLayerSource or nameLayerSource in listNameLayer):
            isReady = False
            break
        if isReady:
          listNameLayerNotReady.remove(nameLayer)
          listNameLayer.append(nameLayer)
          restartLoop = True # because listNameLayerNotReady is modified
          break
      if restartLoop:
        continue
      break
    if listNameLayerNotReady:
      print ('\nlayer(s) not ready:\n', listNameLayerNotReady)
      print ('\nlayer(s) ready:\n', listNameLayer)
      for nameLayer in listNameLayerNotReady:
        print (nameLayer, self.dictGraph[nameLayer], self.dictGraphReversed[nameLayer])
      self.exitGateway(6)
    self.listNameLayer = listNameLayer

  def validateDimension(self):
    if not self.validateConnection():
      self.exitGateway(7)
    self.computeReversedGraph()
    self.computeLayerPrecedence()
    for nameLayer in self.listNameLayer:
      layer = self.dictLayer[nameLayer]
      dictDimensionInput = {}
      dimensionInputResult = []
      dimensionOutputResult = []
      #################
      # input dimension
      #################
      for nameLayerBackward in self.dictGraphReversed[nameLayer]:
        layerBackward = self.dictLayer[nameLayerBackward]
        dimensionOutputLayerBackward = layerBackward.dimensionOutputResult
        dictDimensionInput[nameLayerBackward] = copy.deepcopy(dimensionOutputLayerBackward)
        if layer._type == 'Conv2d':
          if not dimensionInputResult:
            dimensionInputResult = copy.deepcopy(dimensionOutputLayerBackward)
          else:
            if dimensionInputResult[0] != dimensionOutputLayerBackward[0]:
              self.exitGateway(8)
            if dimensionInputResult[1] != dimensionOutputLayerBackward[1]:
              self.exitGateway(9)
            # expect 'Input'/'Conv2d' and stacking of out_channels
            dimensionInputResult[2] += dimensionOutputLayerBackward[2]
        elif layer._type == 'Linear':
          if not dimensionInputResult:
            dimensionInputResult = [0]
          if layerBackward._type == 'Input':
            if len(dimensionOutputLayerBackward) == 1:
              dimensionInputResult[0] += dimensionOutputLayerBackward[0]
            elif len(dimensionOutputLayerBackward) == 3:
              dimensionInputResult[0] += dimensionOutputLayerBackward[0] * dimensionOutputLayerBackward[1] * dimensionOutputLayerBackward[2]
            else:
              self.exitGateway(10)
          elif layerBackward._type == 'Conv2d' or layerBackward._type == 'ConvTranspose2d':
            dimensionInputResult[0] += dimensionOutputLayerBackward[0] * dimensionOutputLayerBackward[1] * dimensionOutputLayerBackward[2]
          elif layerBackward._type == 'Linear':
            dimensionInputResult[0] += dimensionOutputLayerBackward[0]
        elif layer._type == 'ConvTranspose2d':
          if not dimensionInputResult:
            dimensionInputResult = copy.deepcopy(layer.dimensionInputResult)
            dimensionInputResult[2] = 0
          if len(dimensionOutputLayerBackward) == 3:
            if dimensionInputResult[0] != dimensionOutputLayerBackward[0]:
              self.exitGateway(11)
            if dimensionInputResult[1] != dimensionOutputLayerBackward[1]:
              self.exitGateway(12)
            # expect 'Input'/'ConvTranspose2d' and stacking of out_channels
            dimensionInputResult[2] += dimensionOutputLayerBackward[2]
          else: # Linear / Input (1D)
            if dimensionInputResult[0] * dimensionInputResult[1] > dimensionOutputLayerBackward[0]:
              self.exitGateway(13)
            depthInput = dimensionOutputLayerBackward[0] / (layer.dimensionInputResult[0] * layer.dimensionInputResult[1])
            if not depthInput.is_integer():
              self.exitGateway(14)
            dimensionInputResult[2] += int(depthInput)
        self.dictLayer[nameLayer].dictDimensionInput = copy.deepcopy(dictDimensionInput)
        self.dictLayer[nameLayer].dimensionInputResult = copy.deepcopy(dimensionInputResult)
      ##################
      # output dimension
      ##################
      if layer._type == 'Conv2d' or layer._type == 'ConvTranspose2d':
        in_channels = layer.dimension[0]
        out_channels = layer.dimension[1]
        kernel_size = layer.dimension[2]
        stride = layer.dimension[3]
        padding = layer.dimension[4]
        if in_channels != dimensionInputResult[2]:
          self.exitGateway(15)
        if kernel_size > dimensionInputResult[0] or kernel_size > dimensionInputResult[1]:
          if layer._type == 'Conv2d':
            self.exitGateway(16)
        if stride > kernel_size:
          self.exitGateway(17)
        widthOutput = int((dimensionInputResult[0] + 2 * padding - kernel_size) / stride) + 1
        heightOutput = int((dimensionInputResult[1] + 2 * padding - kernel_size) / stride) + 1
        if layer._type == 'ConvTranspose2d':
          widthOutput = int((dimensionInputResult[0] - 1) * stride) + kernel_size - 2 * padding
          heightOutput = int((dimensionInputResult[1] - 1) * stride) + kernel_size - 2 * padding
        depthOutput = out_channels
        self.dictLayer[nameLayer].dimensionOutputResult = [widthOutput, heightOutput, depthOutput]
      elif layer._type == 'Linear':
        in_features = layer.dimension[0]
        out_features = layer.dimension[1]
        if in_features != dimensionInputResult[0]:
          self.exitGateway(18)
        self.dictLayer[nameLayer].dimensionOutputResult = [out_features]
    for nameOutput in self.listNameOutput:
      for nameLayerBackward in self.dictGraphReversed[nameOutput]:
        layerBackward = self.dictLayer[nameLayerBackward]
        dimensionOutputLayerBackward = layerBackward.dimensionOutputResult
        if not dimensionOutputLayerBackward == self.dictLayer[nameOutput].dimensionInputResult:
          self.exitGateway(19)
    print ('\n@@@ all dimension are valid @@@')
    return True

  def validateGraph(self):
    return self.validateDimension()

  def getLayerName(self, positionLayer):
    listLayer = self.listNameInput + self.listNameLayer
    numberLayer = len(listLayer)
    indexLayer = min(int(positionLayer * numberLayer), numberLayer - 1)
    return listLayer[indexLayer]

  def getLayerDimensionC2C(self, dimensionInput, dimensionOutput, numFilter, sizeFilter, reverse = False):
    sizeInput = dimensionInput[0]
    sizeOutput = dimensionOutput[0]
    listSizeFeasible = []
    if sizeInput == sizeOutput:
      if dimensionOutput[0] >= 3:
        listSizeFeasible.append(3)
      listSizeFeasible.append(1)
    else:
      for strideFeasible in range(1, sizeInput):
        sizeFilterFeasible = sizeInput - strideFeasible * (sizeOutput - 1)
        if sizeFilterFeasible < len(listSizeFeasible) + 1:
          break
        listSizeFeasible.append(sizeFilterFeasible)
    inChannel = dimensionInput[2]
    outChannel = max(1, int(numFilter * dimensionOutput[2]))
    sizeKernel = listSizeFeasible[min(int(sizeFilter * len(listSizeFeasible)), len(listSizeFeasible) - 1)]
    stride = min(int(sizeFilter * len(listSizeFeasible)), len(listSizeFeasible) - 1) + 1
    padding = 0
    if sizeInput == sizeOutput:
      if sizeKernel == 1 or sizeKernel == 3:
        stride = 1
      if sizeKernel == 3:
        padding = 1
    if reverse:
      inChannel = dimensionOutput[2]
      outChannel = max(1, int(numFilter * dimensionInput[2]))
    return [inChannel, outChannel, sizeKernel, stride, padding]

  def getLayerDimensionC2L(self, dimensionInput, dimensionOutput, sizeLinear):
    sizeInput = dimensionInput[0]
    sizeOutput = dimensionOutput[0]
    # initialize as if no feasible Conv2d filter size, create equivalent Linear layer
    inChannel = dimensionInput[2]
    outChannel = max(1, int(sizeLinear * sizeOutput))
    sizeKernel = sizeInput
    stride = sizeInput
    padding = 0
    # find all feasible size for square root of (output divided by depth)
    listSqrtSizeFeasible = []
    for depth in range(1, sizeOutput):
      modulusSize = sizeOutput % depth
      if modulusSize == 0:
        sqrtSizeFeasible = (sizeOutput / depth) ** (0.5)
        if sqrtSizeFeasible.is_integer() and sqrtSizeFeasible <= sizeInput:
          if sqrtSizeFeasible < 1:
            break
          listSqrtSizeFeasible.append(int(sqrtSizeFeasible))
    if len(listSqrtSizeFeasible) > 0:
      dictSizeFilterFeasible = {}
      for sqrtSizeFeasible in listSqrtSizeFeasible:
        listSizeFilterFeasible = []
        for strideFeasible in range(1, sizeInput):
          sizeFilterFeasible = sizeInput - strideFeasible * (sqrtSizeFeasible - 1)
          if sizeFilterFeasible < len(listSizeFilterFeasible) + 1:
            break
          listSizeFilterFeasible.append(sizeFilterFeasible)
        dictSizeFilterFeasible[sqrtSizeFeasible] = listSizeFilterFeasible
      # larger the sqrtSizeOutput, the lesser number of parameters in new layer
      sqrtSizeOutput = listSqrtSizeFeasible[min(int(sizeLinear * len(listSqrtSizeFeasible)), len(listSqrtSizeFeasible) - 1)]
      # smaller the indexSizeFilter, the smaller Conv2d filter size
      indexSizeFilter = min(int(sizeFilter * len(dictSizeFilterFeasible[sqrtSizeOutput])), len(dictSizeFilterFeasible[sqrtSizeOutput]) - 1)
      inChannel = dimensionInput[2]
      outChannelMax = int(sizeOutput / (sqrtSizeOutput * sqrtSizeOutput))
      outChannel = max(1, int(sizeLinear * outChannelMax))
      sizeKernel = dictSizeFilterFeasible[sqrtSizeOutput][indexSizeFilter]
      stride = indexSizeFilter + 1
    return [inChannel, outChannel, sizeKernel, stride, padding]

  def getLayerDimensionL2L(self, dimensionInput, dimensionOutput, sizeLinear):
    sizeInput = dimensionInput[0]
    sizeOutput = dimensionOutput[0]
    sizeOutputNew = max(1, int(sizeLinear * sizeOutput))
    return [sizeInput, sizeOutputNew]

  def addLayer(self, layerStart, layerEnd, typeLayer, dimensionLayer):
    totalLayer = len(self.listNameLayer)
    nameLayer = 'layer' + str(totalLayer)
    while nameLayer in self.listNameLayer:
      totalLayer += 1
      nameLayer = 'layer' + str(totalLayer)
    nameLayerStart = layerStart.name
    nameLayerEnd = layerEnd.name
    dimensionInput = layerStart.dimensionOutputResult
    dimensionOutput = layerEnd.dimensionInputResult
    dimensionInputNew = []
    if typeLayer == 'ConvTranspose2d':
      if layerStart._type == 'Linear' or (layerStart._type == 'Input' and len(layerStart.dimensionOutputResult) == 1):
        widthNew = (dimensionInput[0] / dimensionLayer[0]) ** 0.5
        heightNew = (dimensionInput[0] / dimensionLayer[0]) ** 0.5
        dimensionInputNew = [int(widthNew), int(heightNew), dimensionLayer[0]]
      elif layerStart._type == 'ConvTranspose2d' or (layerStart._type == 'Input' and len(layerStart.dimensionOutputResult) == 3):
        dimensionInputNew = copy.deepcopy(dimensionInput)
    layerNew = _layer(nameLayer, typeLayer, dimensionLayer, self.birthdayLatest + 1, dimensionInputNew)
    self.birthdayLatest += 1
    self.listNameLayer.append(nameLayer)
    self.dictLayer[nameLayer] = layerNew
    self.dictGraph[nameLayerStart].append(nameLayer)
    self.dictGraph[nameLayer] = [nameLayerEnd]
    self.computeLayerPrecedence()
    dimensionExtra = [copy.deepcopy(layerEnd.dimension[0])]
    if typeLayer == 'Conv2d':
      if layerEnd._type == 'Conv2d':
        layerEnd.dimension[0] += dimensionLayer[1]
      elif layerEnd._type == 'Linear':
        sizeKernel = dimensionLayer[2]
        stride = dimensionLayer[3]
        padding = dimensionLayer[4]
        widthOutput = int((layerStart.dimensionOutputResult[0] - sizeKernel + 2 * padding) / stride) + 1
        heightOutput = int((layerStart.dimensionOutputResult[1] - sizeKernel + 2 * padding) / stride) + 1
        depthOutput = dimensionLayer[1]
        layerEnd.dimension[0] += widthOutput * heightOutput * depthOutput
    elif typeLayer == 'Linear':
      if layerEnd._type == 'Linear':
        layerEnd.dimension[0] += dimensionLayer[1]
      elif layerEnd._type == 'ConvTranspose2d':
        layerEnd.dimension[0] += int(dimensionLayer[1] / (dimensionOutput[0] * dimensionOutput[1]))
    elif typeLayer == 'ConvTranspose2d':
      if layerEnd._type == 'ConvTranspose2d':
        layerEnd.dimension[0] += dimensionLayer[1]
      elif layerEnd._type == 'Linear':
        sizeKernel = dimensionLayer[2]
        stride = dimensionLayer[3]
        padding = dimensionLayer[4]
        widthOutput = int((dimensionInputNew[0] - 1) * stride) - 2 * padding + sizeKernel
        heightOutput = int((dimensionInputNew[1] - 1) * stride) - 2 * padding + sizeKernel
        depthOutput = dimensionLayer[1]
        layerEnd.dimension[0] += widthOutput * heightOutput * depthOutput
    dimensionExtra.append(copy.deepcopy(layerEnd.dimension[0]) - 1)
    return dimensionExtra

  def actionAdd(self, positionStart, positionEnd, sizeFilter, numFilter, sizeLinear):
    self.actionOnModel = 'None'
    self.dictDiffLayer = {}
    print ()
    print ('##########')
    print ('ACTION ADD')
    print ('##########')
    # TODO implement recurrent connection
    if positionStart > positionEnd:
      return False
    nameLayerStart = self.getLayerName(positionStart)
    nameLayerEnd = self.getLayerName(positionEnd)
    # TODO allow adding parallel layer
    if nameLayerStart == nameLayerEnd:
      return False
    layerStart = self.dictLayer[nameLayerStart]
    layerEnd = self.dictLayer[nameLayerEnd]
    dimensionInput = layerStart.dimensionOutputResult
    dimensionOutput = layerEnd.dimensionInputResult
    typeLayer = ''
    dimensionLayer = []
    if ((layerStart._type == 'Input' and len(dimensionInput) == 3) or layerStart._type == 'Conv2d') and layerEnd._type == 'Conv2d':
      if dimensionInput[0] < dimensionOutput[0]:
        return False
      print ()
      print ('################')
      print ('Conv2d to Conv2d')
      print ('################')
      typeLayer = 'Conv2d'
      dimensionLayer = self.getLayerDimensionC2C(dimensionInput, dimensionOutput, numFilter, sizeFilter)
    elif ((layerStart._type == 'Input' and len(dimensionInput) == 3) or layerStart._type == 'Conv2d') and layerEnd._type == 'Linear':
      print ()
      print ('################')
      print ('Conv2d to Linear')
      print ('################')
      typeLayer = 'Conv2d'
      dimensionLayer = self.getLayerDimensionC2L(dimensionInput, dimensionOutput, sizeLinear)
    elif ((layerStart._type == 'Input' and len(dimensionInput) == 1) or layerStart._type == 'Linear') and layerEnd._type == 'Linear':
      print ()
      print ('################')
      print ('Linear to Linear')
      print ('################')
      typeLayer = 'Linear'
      dimensionLayer = self.getLayerDimensionL2L(dimensionInput, dimensionOutput, sizeLinear)
    elif ((layerStart._type == 'Input' and len(dimensionInput) == 1) or layerStart._type == 'Linear') and layerEnd._type == 'ConvTranspose2d':
      if dimensionInput[0] < dimensionOutput[0] * dimensionOutput[1]:
        return False
      listSqrtSizeFeasible = []
      for depth in range(1, dimensionInput[0]):
        modulusSize = dimensionInput[0] % depth
        if modulusSize == 0:
          sqrtSizeFeasible = (dimensionInput[0] / depth) ** (0.5)
          if sqrtSizeFeasible.is_integer() and (sqrtSizeFeasible <= dimensionOutput[0] or sqrtSizeFeasible <= dimensionOutput[1]):
            if sqrtSizeFeasible < 1:
              break
            listSqrtSizeFeasible.append(int(sqrtSizeFeasible))
      if not listSqrtSizeFeasible:
        return False
      sizeOutput = listSqrtSizeFeasible[min(int(sizeLinear * len(listSqrtSizeFeasible)), len(listSqrtSizeFeasible) - 1)]
      dimensionOutputNew = [sizeOutput, sizeOutput, int(dimensionInput[0] / (sizeOutput * sizeOutput))]
      print ()
      print ('#########################')
      print ('Linear to ConvTranspose2d')
      print ('#########################')
      typeLayer = 'ConvTranspose2d'
      dimensionLayer = self.getLayerDimensionC2C(dimensionOutput, dimensionOutputNew, numFilter, sizeFilter, True)
    elif ((layerStart._type == 'Input' and len(dimensionInput) == 3) or layerStart._type == 'ConvTranspose2d') and layerEnd._type == 'ConvTranspose2d':
      if dimensionInput[0] > dimensionOutput[0]:
        return False
      print ()
      print ('##################################')
      print ('ConvTranspose2d to ConvTranspose2d')
      print ('##################################')
      typeLayer = 'ConvTranspose2d'
      dimensionLayer = self.getLayerDimensionC2C(dimensionOutput, dimensionInput, numFilter, sizeFilter, True)
    else:
      return False
    dimensionExtra = self.addLayer(layerStart, layerEnd, typeLayer, dimensionLayer)
    print ('\n@@@ graph is valid:', self.validateGraph(), '@@@')
    self.actionOnModel = 'Add'
    self.dictDiffLayer = {nameLayerEnd: [dimensionExtra]}
    return True

  def getPath(self, nameLayerStart, nameLayerEnd):
    # get all path originates from nameLayerStart and ends at nameLayerEnd
    dictPath = {}
    listNameLayerNext = self.dictGraph[nameLayerStart]
    for nameLayerNext in listNameLayerNext:
      dictPath[nameLayerNext] = {}
      if nameLayerNext == nameLayerEnd:
        continue
      elif 'output' in nameLayerNext:
        del dictPath[nameLayerNext]
      else:
        dictPath[nameLayerNext] = self.getPath(nameLayerNext, nameLayerEnd)
        if not dictPath[nameLayerNext]:
          del dictPath[nameLayerNext]
    return dictPath

  def flattenPath(self, dictPath):
    # seperate dictPath into list of path
    listPath = []
    if not dictPath:
      listPath.append([])
    else:
      for nameLayer in dictPath:
        listPathBranch = self.flattenPath(dictPath[nameLayer])
        for pathBranch in listPathBranch:
          listPath.append([nameLayer] + pathBranch)
    return listPath

  def filterPath(self, listPath):
    # filter listPath which contains layer(s) of I/O's only path
    listNameLayer = []
    for nameInput in self.listNameInput:
      listNext = self.dictGraph[nameInput]
      while len(listNext) == 1:
        nameNext = listNext[0]
        if 'output' in nameNext:
          break
        listNameLayer.append(nameNext)
        listNext = self.dictGraph[nameNext]
    for nameOutput in self.listNameOutput:
      listPrevious = self.dictGraphReversed[nameOutput]
      while len(listPrevious) == 1:
        namePrevious = listPrevious[0]
        if 'input' in namePrevious:
          break
        listNameLayer.append(namePrevious)
        listPrevious = self.dictGraphReversed[namePrevious]
    listPathValid = []
    for path in listPath:
      isPathClear = True
      for nameLayer in path:
        if nameLayer in listNameLayer:
          isPathClear = False
          break
      if isPathClear:
        listPathValid.append(path)
    return listPathValid

  def getDeadEnd(self, dictGraphOriginal, dictPath):
    # compute resultant graph (stand-alone) after path(s) deduction
    # find isolated layer(s)
    # mark trimmed path(s)
    dictGraph = copy.deepcopy(dictGraphOriginal)
    listIsolated = []
    countIsolated = -1
    while len(listIsolated) != countIsolated:
      countIsolated = len(listIsolated)
      # trim path in dictGraph
      dictGraphCopy = copy.deepcopy(dictGraph)
      for nameLayer in dictGraphCopy:
        for nameIsolated in listIsolated:
          if nameIsolated in dictGraphCopy[nameLayer]:
            if nameLayer not in dictPath:
              dictPath[nameLayer] = [nameIsolated]
            else:
              if nameIsolated not in dictPath[nameLayer]:
                dictPath[nameLayer].append(nameIsolated)
            dictGraph[nameLayer].remove(nameIsolated)
      # locate isolated layer (in dictGraph)
      for nameLayer in dictGraph:
        if len(dictGraph[nameLayer]) == 0 and nameLayer not in listIsolated:
          listIsolated.append(nameLayer)
      # early continuing such that all isolated layers are found from dictGraph
      if countIsolated != len(listIsolated):
        continue
      # locate isolated layer (in dictReversed)
      dictReversed = self.computeReversedGraphAlone(dictGraph)
      for nameLayer in dictReversed:
        if len(dictReversed[nameLayer]) == 0 and nameLayer not in listIsolated:
          listIsolated.append(nameLayer)
          if nameLayer not in dictPath:
            dictPath[nameLayer] = copy.deepcopy(dictGraph[nameLayer])
            dictGraph[nameLayer] = []
          else:
            for nameNext in dictGraphCopy[nameLayer]:
              if nameNext not in dictPath[nameLayer]:
                dictPath[nameLayer].append(nameNext)
                dictGraph[nameLayer].remove(nameNext)
    return dictGraph, listIsolated, dictPath

  def removePath(self, path):
    dictGraph = copy.deepcopy(self.dictGraph)
    listIsolated = []
    dictPath = {}
    for number in range(len(path) - 1):
      nameThis = path[number]
      nameNext = path[number + 1]
      dictGraph[nameThis].remove(nameNext)
      # TODO extract position in the dictGraph[nameThis]
      if nameThis not in dictPath:
        dictPath[nameThis] = [nameNext]
      else:
        dictPath[nameThis].append(nameNext)
    dictGraph, listIsolated, dictPath = self.getDeadEnd(dictGraph, dictPath)
    # dictReversed = self.computeReversedGraphAlone(dictGraph)
    # obtain layers' begining positions and dimensions for removal
    dictPathTrimed = {}
    dictGraphReversedSorted = {}
    for nameThis in dictPath:
      layerThis = self.dictLayer[nameThis]
      typeThis = layerThis._type
      dimensionThis = layerThis.dimension
      outputThis = layerThis.dimensionOutputResult
      listNext = dictPath[nameThis]
      for nameNext in listNext:
        if nameNext not in listIsolated:
          layerNext = self.dictLayer[nameNext]
          typeNext = layerNext._type
          dimensionNext = layerNext.dimension
          if nameNext not in dictPathTrimed:
            dictPathTrimed[nameNext] = []
          # begining position for removal
          dimensionStart = 0
          # changing graph may affect the precedence of layers
          if nameNext not in dictGraphReversedSorted:
            listReversed = copy.deepcopy(self.dictGraphReversed[nameNext])
            listReversed.sort(key = lambda item : self.dictLayer[item].birthday)
            dictGraphReversedSorted[nameNext] = listReversed
          for namePast in dictGraphReversedSorted[nameNext]:
            if namePast == nameThis:
              break
            else:
              layerPast = self.dictLayer[namePast]
              typePast = layerPast._type
              dimensionPast = layerPast.dimension
              outputPast = layerPast.dimensionOutputResult
              if typeNext == 'Conv2d':
                dimensionStart += dimensionPast[1]
              elif typeNext == 'Linear':
                if typePast == 'Conv2d' or (typePast == 'Input' and len(outputPast) == 3):
                  dimensionStart += outputPast[0] * outputPast[1] * outputPast[2]
                elif typePast == 'Linear' or (typePast == 'Input' and len(outputPast) == 1):
                  dimensionStart += dimensionPast[1]
              elif typeNext == 'ConvTranspose2d':
                if typePast == 'ConvTranspose2d' or (typePast == 'Input' and len(outputPast) == 3):
                  dimensionStart += dimensionPast[1]
                elif typePast == 'Linear' or (typePast == 'Input' and len(outputPast) == 1):
                  dimensionInputNext = layerNext.dimensionInputResult
                  dimensionStart += int(dimensionPast[1] / (dimensionInputNext[0] * dimensionInputNext[1]))
          # dimension for removal
          dimensionTrimed = 0
          if typeNext == 'Conv2d':
            dimensionTrimed = dimensionThis[1]
          elif typeNext == 'Linear':
            if typeThis == 'Conv2d' or (typeThis == 'Input' and len(outputThis) == 3):
              dimensionTrimed = outputThis[0] * outputThis[1] * outputThis[2]
            elif typeThis == 'Linear' or (typeThis == 'Input' and len(outputThis) == 1):
              dimensionTrimed = dimensionThis[1]
          elif typeNext == 'ConvTranspose2d':
            if typeThis == 'ConvTranspose2d' or (typeThis == 'Input' and len(outputThis) == 3):
              dimensionTrimed = dimensionThis[1]
            elif typeThis == 'Linear' or (typeThis == 'Input' and len(outputThis) == 1):
              dimensionInputNext = layerNext.dimensionInputResult
              dimensionTrimed = int(dimensionThis[1] / (dimensionInputNext[0] * dimensionInputNext[1]))
          listDimensionTrimed = [dimensionStart, dimensionStart + dimensionTrimed]
          dictPathTrimed[nameNext].append(listDimensionTrimed)
    # sort and merge trimed path's dimension
    for nameThis in dictPathTrimed:
      dictPathTrimed[nameThis].sort(key = lambda item : item[0])
      pathAll = []
      for pathTrimed in dictPathTrimed[nameThis]:
        for position in pathTrimed:
          if len(pathAll) > 0:
            if pathAll[-1] != position:
              pathAll.append(position)
            else:
              pathAll.pop()
          else:
            pathAll.append(position)
      dictPathTrimed[nameThis] = []
      for counter in range(len(pathAll)):
        if counter % 2 == 1:
          dictPathTrimed[nameThis].append([pathAll[counter - 1], pathAll[counter]])
    # modify layers' dimension
    for nameThis in dictPath:
      layerThis = self.dictLayer[nameThis]
      typeThis = layerThis._type
      dimensionThis = layerThis.dimension
      outputThis = layerThis.dimensionOutputResult
      listNext = dictPath[nameThis]
      for nameNext in listNext:
        layerNext = self.dictLayer[nameNext]
        typeNext = layerNext._type
        dimensionNext = layerNext.dimension
        if typeNext == 'Conv2d':
          dimensionNext[0] -= dimensionThis[1]
        elif typeNext == 'Linear':
          if typeThis == 'Conv2d' or (typeThis == 'Input' and len(outputThis) == 3):
            dimensionNext[0] -= outputThis[0] * outputThis[1] * outputThis[2]
          elif typeThis == 'Linear' or (typeThis == 'Input' and len(outputThis) == 1):
            dimensionNext[0] -= dimensionThis[1]
        if typeNext == 'ConvTranspose2d':
          if typeThis == 'ConvTranspose2d' or (typeThis == 'Input' and len(outputThis) == 3):
            dimensionNext[0] -= dimensionThis[1]
          elif typeThis == 'Linear' or (typeThis == 'Input' and len(outputThis) == 1):
            dimensionInputNext = layerNext.dimensionInputResult
            dimensionNext[0] -= int(dimensionThis[1] / (dimensionInputNext[0] * dimensionInputNext[1]))
    # trim layer information
    self.dictGraph = copy.deepcopy(dictGraph)
    for nameIsolated in listIsolated:
      del self.dictGraph[nameIsolated]
      del self.dictLayer[nameIsolated]
      self.listNameLayer.remove(nameIsolated)
      dictPathTrimed[nameIsolated] = []
    return dictPathTrimed

  def actionRemove(self, positionStart, positionEnd, lengthPath):
    self.actionOnModel = 'None'
    self.dictDiffLayer = {}
    print ()
    print ('#############')
    print ('ACTION REMOVE')
    print ('#############')
    # TODO implement recurrent connection
    if positionStart > positionEnd:
      return False
    nameLayerStart = self.getLayerName(positionStart)
    nameLayerEnd = self.getLayerName(positionEnd)
    if nameLayerStart == nameLayerEnd:
      return False
    if 'input' in nameLayerEnd:
      return False
    # pre-requisite for more than 1 distinct path
    if not (len(self.dictGraph[nameLayerStart]) > 1 and len(self.dictGraphReversed[nameLayerEnd]) > 1):
      return False
    dictPath = self.getPath(nameLayerStart, nameLayerEnd)
    # there are less than 2 distinct paths
    if len(dictPath) < 2:
      return False
    listPath = self.flattenPath(dictPath)
    listPathValid = self.filterPath(listPath)
    if len(listPathValid) == 0:
      return False
    listPathValid.sort(key = len)
    indexPath = min(int(lengthPath * len(listPathValid)), len(listPathValid) - 1)
    pathRemove = listPathValid[indexPath]
    pathRemove = [nameLayerStart] + pathRemove
    dictPathTrimed = self.removePath(pathRemove)
    print ('\n@@@ graph is valid:', self.validateGraph(), '@@@')
    self.actionOnModel = 'Remove'
    self.dictDiffLayer = dictPathTrimed
    return True

  def createPyTorchScript(self):
    self.computeReversedGraph()
    self.computeLayerPrecedence()
    scriptLibary = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
    scriptClass = '\nclass _net(nn.Module):'
    # initialize
    scriptInit = '\n  def __init__(self):\n    super(_net, self).__init__()'
    for nameLayer in self.listNameLayer:
      layer = self.dictLayer[nameLayer]
      scriptInit += '\n    self.' + layer.name + ' = nn.' + layer._type
      scriptDimension = ''
      for dimension in layer.dimension:
        scriptDimension += ', ' + str(dimension)
      scriptInit += '(' + scriptDimension[2:] + ').double()'
    # inference
    scriptForward = '\n\n  def forward(self, inputModel):'
    for countInput in range(len(self.listNameInput)):
      scriptForward += '\n    ' + self.listNameInput[countInput] + ' = inputModel[' + str(countInput) + ']'
    for nameLayer in self.listNameLayer:
      typeLayer = self.dictLayer[nameLayer]._type
      listNameSource = self.dictGraphReversed[nameLayer]
      strInput = ''
      for nameSource in listNameSource:
        dimensionOutput = self.dictLayer[nameSource].dimensionOutputResult
        strInput += ', '
        if len(dimensionOutput) == 3 and typeLayer == 'Linear':
          dimensionResult = dimensionOutput[0] * dimensionOutput[1] * dimensionOutput[2]
          if 'input' not in nameSource:
            strInput += 'output_'
          strInput += nameSource + '.view(-1, ' + str(dimensionResult) + ')'
        elif len(dimensionOutput) == 1 and typeLayer == 'ConvTranspose2d':
          dimensionInput = self.dictLayer[nameLayer].dimensionInputResult
          depthInput = int(dimensionOutput[0] / (dimensionInput[0] * dimensionInput[1]))
          if 'input' not in nameSource:
            strInput += 'output_'
          strInput += nameSource + '.view(1, ' + str(depthInput) + ', ' + str(dimensionInput[0]) + ', ' + str(dimensionInput[1]) + ')'
        else:
          if 'input' not in nameSource:
            strInput += 'output_'
          strInput += nameSource
      strInput = strInput[2:]
      if len(listNameSource) > 1:
        strInput = 'torch.cat([' + strInput + '], 1)'
      scriptForward += '\n    output_' + nameLayer + ' = '
      if typeLayer == 'Conv2d':
        scriptForward += 'F.relu(self.'
      elif typeLayer == 'Linear':
        scriptForward += 'torch.sigmoid(self.'
      elif typeLayer == 'ConvTranspose2d':
        scriptForward += 'F.relu(self.'
      scriptForward += nameLayer + '(' + strInput + '))'
    scriptOutput = ''
    for nameOutput in self.listNameOutput:
      scriptOutput += '\n    ' + nameOutput + ' = output_' + self.dictGraphReversed[nameOutput][0]
    scriptForward += scriptOutput
    # return
    scriptReturn = ''
    for nameOutput in self.listNameOutput:
      scriptReturn += ', ' + nameOutput
    scriptReturn = '\n    return [' + scriptReturn[2:] + ']'
    script = scriptLibary + scriptClass + scriptInit + scriptForward + scriptReturn
    return script

  def modifyModelAdd(self):
    stateDict = self.modelNext.state_dict()
    for key in self.modelNow.state_dict():
      nameLayer = key.split('.')[0]
      if 'weight' in key and nameLayer in self.dictDiffLayer:
        paraOld = self.modelNow.state_dict()[key].cpu().data.numpy()
        paraNew = self.modelNext.state_dict()[key].cpu().data.numpy()
        # TODO initialize with small values instead of zeros
        paraNew = np.zeros(paraNew.shape)
        typeLayer = mk1.dictLayer[nameLayer]._type
        if typeLayer == 'Conv2d':
          paraNew[:, : self.dictDiffLayer[nameLayer][0][0], :, :] = paraOld
        elif typeLayer == 'Linear':
          paraNew[:, : self.dictDiffLayer[nameLayer][0][0]] = paraOld
        elif typeLayer == 'ConvTranspose2d':
          paraNew[: self.dictDiffLayer[nameLayer][0][0], :, :, :] = paraOld
        stateDict[key] = torch.tensor(paraNew).cuda()
      else:
        stateDict[key] = self.modelNow.state_dict()[key]
    self.modelNext.load_state_dict(stateDict)

  def modifyModelRemove(self):
    stateDictNew = self.modelNext.state_dict()
    stateDictOld = self.modelNow.state_dict()
    for key in self.modelNow.state_dict():
      nameLayer = key.split('.')[0]
      if nameLayer in self.dictDiffLayer:
        if 'weight' in key and len(self.dictDiffLayer[nameLayer]) > 0:
          dimensionTrimed = self.dictDiffLayer[nameLayer]
          typeLayer = mk1.dictLayer[nameLayer]._type
          paraOld = self.modelNow.state_dict()[key].cpu().data.numpy()
          paraNew = self.modelNext.state_dict()[key].cpu().data.numpy()
          for pathTrimed in dimensionTrimed:
            # TODO gradually replace parameters with zeros
            paraZeros = np.zeros(paraOld.shape)
            if typeLayer == 'Conv2d':
              paraOld[:, pathTrimed[0] : pathTrimed[1], :, :] = paraZeros[:, pathTrimed[0] : pathTrimed[1], :, :]
            elif typeLayer == 'Linear':
              paraOld[:, pathTrimed[0] : pathTrimed[1]] = paraZeros[:, pathTrimed[0] : pathTrimed[1]]
            elif typeLayer == 'ConvTranspose2d':
              paraOld[pathTrimed[0] : pathTrimed[1], :, :, :] = paraZeros[pathTrimed[0] : pathTrimed[1], :, :, :]
            stateDictOld[key] = torch.tensor(paraOld).cuda()
          startOld = 0
          endOld = 0
          startNew = 0
          endNew = 0
          for interval in range(len(dimensionTrimed) + 1):
            if interval == 0:
              if dimensionTrimed[0][0] != 0:
                startOld = 0
                endOld = dimensionTrimed[0][0]
                startNew = 0
                endNew = dimensionTrimed[0][0]
              else:
                continue
            elif interval == len(dimensionTrimed):
              dimensionEnd = 1
              if typeLayer == 'ConvTranspose2d':
                dimensionEnd = 0
              if dimensionTrimed[interval - 1][1] != paraOld.shape[dimensionEnd]:
                startOld = dimensionTrimed[len(dimensionTrimed) - 1][1]
                endOld = paraOld.shape[dimensionEnd]
                startNew = endNew
                endNew = paraNew.shape[dimensionEnd]
              else:
                continue
            else:
              startOld = dimensionTrimed[interval - 1][1]
              endOld = dimensionTrimed[interval][0]
              startNew = endNew
              endNew = startNew + dimensionTrimed[interval][0] - dimensionTrimed[interval - 1][1]
            if typeLayer == 'Conv2d':
              paraNew[:, startNew : endNew, :, :] = paraOld[:, startOld : endOld, :, :]
            elif typeLayer == 'Linear':
              paraNew[:, startNew : endNew] = paraOld[:, startOld : endOld]
            elif typeLayer == 'ConvTranspose2d':
              paraNew[startNew : endNew, :, :, :] = paraOld[startOld : endOld, :, :, :]
            stateDictNew[key] = torch.tensor(paraNew).cuda()
        elif 'bias' in key and len(self.dictDiffLayer[nameLayer]) > 0:
          stateDictNew[key] = self.modelNow.state_dict()[key]
      else:
        stateDictNew[key] = self.modelNow.state_dict()[key]
    self.modelNow.load_state_dict(stateDictOld)
    self.modelNext.load_state_dict(stateDictNew)

  def fillWithValues(self):
    # after training (replaces zeros with small values)
    stateDictNew = self.modelNext.state_dict()
    for key in self.modelNext.state_dict():
      nameLayer = key.split('.')[0]
      if 'weight' in key and nameLayer in self.dictDiffLayer:
        paraNew = self.modelNext.state_dict()[key].cpu().data.numpy()
        bound = 1.0
        if mk1.dictLayer[nameLayer]._type == 'Conv2d':
          bound = 1.0 / ((paraNew.shape[1] * paraNew.shape[2] * paraNew.shape[3]) ** 0.5)
        elif mk1.dictLayer[nameLayer]._type == 'Linear':
          bound = 1.0 / (paraNew.shape[1] ** 0.5)
        elif mk1.dictLayer[nameLayer]._type == 'ConvTranspose2d':
          bound = 1.0 / ((paraNew.shape[1] * paraNew.shape[2] * paraNew.shape[3]) ** 0.5)
        paraInit = np.random.uniform(-bound, bound, paraNew.shape)
        for pathInit in self.dictDiffLayer[nameLayer]:
          if mk1.dictLayer[nameLayer]._type == 'Conv2d':
            paraNew[:, pathInit[0] : pathInit[1], :, :] = paraInit[:, pathInit[0] : pathInit[1], :, :]
          elif mk1.dictLayer[nameLayer]._type == 'Linear':
            paraNew[:, pathInit[0] : pathInit[1]] = paraInit[:, pathInit[0] : pathInit[1]]
          elif mk1.dictLayer[nameLayer]._type == 'ConvTranspose2d':
            paraNew[pathInit[0] : pathInit[1], :, :, :] = paraInit[pathInit[0] : pathInit[1], :, :, :]
        stateDictNew[key] = torch.tensor(paraNew).cuda()
      self.modelNext.load_state_dict(stateDictNew)

# Conv2d -> Conv2d -> Linear -> Linear -> ConvTranspose2d -> ConvTranspose2d
input1 = _layer('input1', 'Input', [120, 120, 3])
input2 = _layer('input2', 'Input', [32])
layer1 = _layer('layer1', 'Conv2d', [3, 32, 3, 2, 1])
layer2 = _layer('layer2', 'Conv2d', [32, 8, 4, 3, 0])
layer3 = _layer('layer3', 'Linear', [2888 + 32, 128])
layer4 = _layer('layer4', 'Linear', [128, 256])
layer5 = _layer('layer5', 'ConvTranspose2d', [4, 8, 6, 4, 1], 0, [8, 8, 4])
layer6 = _layer('layer6', 'ConvTranspose2d', [8, 3, 27, 3, 0], 0, [32, 32, 8])
output1 = _layer('output1', 'Output', [120, 120, 3])
output2 = _layer('output2', 'Output', [256])

listNameInput = ['input1', 'input2']
listNameOutput = ['output1', 'output2']
listLayerName = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']
dictLayer = {'input1': input1, 'input2': input2, 'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4, 'layer5': layer5, 'layer6': layer6, 'output1': output1, 'output2': output2}
dictGraph = {'input1': ['layer1'], 'input2': ['layer3'], 'layer1': ['layer2'], 'layer2': ['layer3'], 'layer3': ['layer4'], 'layer4': ['layer5', 'output2'], 'layer5': ['layer6'], 'layer6': ['output1']}

mk1 = _net('mk1', listNameInput, listNameOutput, listLayerName, dictLayer, dictGraph)

print ('\n@@@ graph is valid:', mk1.validateGraph(), '@@@')
script = mk1.createPyTorchScript()
fileScript = open('net.py', 'w')
fileScript.write(script)
fileScript.close()
import net
imp.reload(net)
mk1.modelNow = net._net()
mk1.modelNext = mk1.modelNow

historyAction = []
countAddNotEqual = 0
countRemoveNotEqual = 0

while True:
  print ()
  print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
  print ('@@@@@@@@@@ Next Cycle @@@@@@@@@@')
  print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
  countAdd = 0
  countRemove = 0
  for counter in range(len(historyAction)):
    if historyAction[counter] == 'Add':
      countAdd += 1
    elif historyAction[counter] == 'Remove':
      countRemove += 1
  print ('Add:', countAdd, 'Not Equal:', countAddNotEqual)
  print ('Remove:', countRemove, 'Not Equal:', countRemoveNotEqual)
  # action
  tensorAction = []
  for counter in range(16):
    tensorAction.append(random.random())
  actionAdd = tensorAction[0]
  actionRemove = tensorAction[1]
  actionRead = tensorAction[2]
  isValidAction = False
  # for debug purpose
  strAction = '\n\n#dictLayer = {'
  for nameLayer in mk1.listNameLayer:
    strAction += ', \'' + nameLayer + '\': \'' + mk1.dictLayer[nameLayer]._type + '\''
  strAction += '}'
  if actionRead > actionAdd and actionRead > actionRemove:
    continue
  if actionAdd > actionRemove and actionAdd > actionRead:
    positionStartAdd = tensorAction[3]
    positionEndAdd = tensorAction[4]
    sizeFilter = tensorAction[5]
    numFilter = tensorAction[6]
    sizeLinear = tensorAction[7]
    isValidAction = mk1.actionAdd(positionStartAdd, positionEndAdd, sizeFilter, numFilter, sizeLinear)
    strAction += '\n\n#ADD\n#' + str(mk1.dictDiffLayer)
  elif actionRemove > actionAdd and actionRemove > actionRead:
    positionStartRemove = tensorAction[8]
    positionEndRemove = tensorAction[9]
    lengthPathRemove = tensorAction[10]
    isValidAction = mk1.actionRemove(positionStartRemove, positionEndRemove, lengthPathRemove)
    strAction += '\n\n#REMOVE\n#' + str(mk1.dictDiffLayer)
  if isValidAction:
    mk1.modelNow = mk1.modelNext
    script = mk1.createPyTorchScript()
    fileScript = open('net.py', 'a')
    fileScript.write(strAction)
    fileScript.close()
    os.rename('net.py', 'netOld.py')
    fileScript = open('net.py', 'w')
    fileScript.write(script)
    fileScript.close()
    import net
    imp.reload(net)
    mk1.modelNext = net._net()
    print ('\ndiffLayer:', mk1.dictDiffLayer)
    if mk1.actionOnModel == 'Add':
      mk1.modifyModelAdd()
    elif mk1.actionOnModel == 'Remove':
      mk1.modifyModelRemove()
    historyAction.append(mk1.actionOnModel)

    # testing output for both models
    mk1.modelNow.cuda()
    mk1.modelNext.cuda()
    inputModel = []
    for nameInput in mk1.listNameInput:
      dimensionInput = mk1.dictLayer[nameInput].dimensionOutputResult
      if len(dimensionInput) == 3:
        inputModel.append(torch.rand(1, dimensionInput[2], dimensionInput[0], dimensionInput[1]).double().cuda())
      elif len(dimensionInput) == 1:
        inputModel.append(torch.rand(1, dimensionInput[0]).double().cuda())
      else:
        print ('undefined input dimension')
        exit()
    tensorOutputOld = mk1.modelNow(inputModel)
    tensorOutputNew = mk1.modelNext(inputModel)
    outputOld = []
    outputNew = []
    for tensorOld in tensorOutputOld:
      outputOld.append(tensorOld.cpu().data.numpy())
    for tensorNew in tensorOutputNew:
      outputNew.append(tensorNew.cpu().data.numpy())
    for countOutput in range(len(outputNew)):
      outNew = outputNew[countOutput]
      outOld = outputOld[countOutput]
      if not np.allclose(outNew, outOld):
        print ('\n@@@ models\' outputs are not equal @@@')
        if actionAdd > actionRemove:
          countAddNotEqual += 1
        elif actionRemove > actionAdd:
          countRemoveNotEqual += 1
        print ('\nold network output:\n', outOld)
        print ('\nnew network output:\n', outNew)
        outOld = outOld.flatten()
        outNew = outNew.flatten()
        countError = 0
        for position in range(len(outNew)):
          if not np.allclose(outNew[position], outOld[position]):
            if countError < 10:
              print ('\nposition:', position, outNew[position], outOld[position], 'error:', np.abs(outNew[position] - outOld[position]))
            countError += 1
        print ('\noutput', str(countOutput + 1), 'total number of error:', countError)
        print ('\naverage error:', np.sum(np.abs(outNew - outOld)) / len(outNew))
        exit()
    print ('\n@@@ models\' outputs are equivalent @@@')

    mk1.fillWithValues()
