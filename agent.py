import copy
import random
import imp
import torch
import numpy as np

historyAction = []

class _layer:
  name = ''
  _type = ''
  dimension = []
  dictDimensionInput = {}
  dimensionInputResult = []
  dimensionOutputResult = []

  def __init__(self, name, _type, dimension):
    self.name = name
    self._type = _type
    self.dimension = dimension
    # conv2d dimension: [in_channels, out_channels, kernel_size, stride, padding]
    # Linear dimension: [in_features, out_features]
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
  # non-essential
  dictGraphReversed = {}

  def __init__(self, name, listNameInput, listNameOutput, listNameLayer, dictLayer, dictGraph):
    self.name = name
    self.listNameInput = listNameInput
    self.listNameOutput = listNameOutput
    self.listNameLayer = listNameLayer
    self.dictLayer = dictLayer
    self.dictGraph = dictGraph

  def exitGateway(self, number):
    print ('\ninput(s):', self.listNameInput)
    print ('\nlayer(s):', self.listNameLayer)
    print ('\noutput(s):', self.listNameOutput)
    print ('\ngraph:\n', self.dictGraph)
    print ('\nreversed graph:\n', self.dictGraphReversed)
    for nameLayer in self.listNameLayer:
      layer = self.dictLayer[nameLayer]
      print ()
      print (nameLayer, layer._type, layer.dimension)
      print (layer.dictDimensionInput)
      print (layer.dimensionInputResult)
      print (layer.dimensionOutputResult)
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
        if not (typeLayerForward == 'Linear' or typeLayerForward == 'Output'):
          self.exitGateway(2)
      elif typeLayer == 'Input':
        if typeLayerForward == 'Output' or not (typeLayerForward == 'Conv2d' or typeLayerForward == 'Linear'):
          self.exitGateway(3)
      else:
        self.exitGateway(4)
    return True

  def validateConnection(self):
    for nameInput in self.listNameInput:
      if not self.hasValidConnection(nameInput):
        self.exitGateway(5)
    for nameLayer in self.listNameLayer:
      if not self.hasValidConnection(nameLayer):
        self.exitGateway(6)
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
      self.exitGateway(7)
    self.listNameLayer = listNameLayer

  def validateDimension(self):
    if not self.validateConnection():
      self.exitGateway(8)
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
              self.exitGateway(9)
            if dimensionInputResult[1] != dimensionOutputLayerBackward[1]:
              self.exitGateway(10)
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
              self.exitGateway(11)
          elif layerBackward._type == 'Conv2d':
            dimensionInputResult[0] += dimensionOutputLayerBackward[0] * dimensionOutputLayerBackward[1] * dimensionOutputLayerBackward[2]
          elif layerBackward._type == 'Linear':
            dimensionInputResult[0] += dimensionOutputLayerBackward[0]
        self.dictLayer[nameLayer].dictDimensionInput = copy.deepcopy(dictDimensionInput)
        self.dictLayer[nameLayer].dimensionInputResult = copy.deepcopy(dimensionInputResult)
      ##################
      # output dimension
      ##################
      if layer._type == 'Conv2d':
        in_channels = layer.dimension[0]
        out_channels = layer.dimension[1]
        kernel_size = layer.dimension[2]
        stride = layer.dimension[3]
        padding = layer.dimension[4]
        if in_channels != dimensionInputResult[2]:
          self.exitGateway(12)
        if kernel_size > dimensionInputResult[0] or kernel_size > dimensionInputResult[1]:
          self.exitGateway(13)
        if stride > kernel_size:
          self.exitGateway(14)
        widthOutput = int((dimensionInputResult[0] + 2 * padding - kernel_size) / stride) + 1
        heightOutput = int((dimensionInputResult[1] + 2 * padding - kernel_size) / stride) + 1
        depthOutput = out_channels
        self.dictLayer[nameLayer].dimensionOutputResult = [widthOutput, heightOutput, depthOutput]
      elif layer._type == 'Linear':
        in_features = layer.dimension[0]
        out_features = layer.dimension[1]
        if in_features != dimensionInputResult[0]:
          self.exitGateway(15)
        self.dictLayer[nameLayer].dimensionOutputResult = [out_features]
    print ('\n@@@ all dimension are valid @@@')
    return True

  def validateGraph(self):
    return self.validateDimension()

  def getLayerName(self, positionLayer):
    listLayer = self.listNameInput + self.listNameLayer
    numberLayer = len(listLayer)
    indexLayer = min(int(positionLayer * numberLayer), numberLayer - 1)
    return listLayer[indexLayer]

  def getLayerDimensionC2C(self, dimensionInput, dimensionOutput, numFilter, sizeFilter):
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
    layerNew = _layer(nameLayer, typeLayer, dimensionLayer)
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
      layerEnd.dimension[0] += dimensionLayer[1]
    dimensionExtra.append(copy.deepcopy(layerEnd.dimension[0]) - 1)
    return dimensionExtra

  def actionAdd(self, positionStart, positionEnd, sizeFilter, numFilter, sizeLinear):
    print ()
    print ('##########')
    print ('ACTION ADD')
    print ('##########')
    # TODO implement recurrent connection
    if positionStart > positionEnd:
      return False, {}
    nameLayerStart = self.getLayerName(positionStart)
    nameLayerEnd = self.getLayerName(positionEnd)
    # TODO allow adding parallel layer
    if nameLayerStart == nameLayerEnd:
      return False, {}
    layerStart = self.dictLayer[nameLayerStart]
    layerEnd = self.dictLayer[nameLayerEnd]
    dimensionInput = layerStart.dimensionOutputResult
    dimensionOutput = layerEnd.dimensionInputResult
    typeLayer = ''
    dimensionLayer = []
    if layerStart._type == 'Input' and len(dimensionInput) == 1 and layerEnd._type == 'Conv2d':
      return False, {}
    if layerEnd._type == 'Input':
      return False, {}
    if layerStart._type == 'Linear' and layerEnd._type == 'Conv2d':
      return False, {}
    if ((layerStart._type == 'Input' and len(dimensionInput) == 3) or layerStart._type == 'Conv2d') and layerEnd._type == 'Conv2d':
      if dimensionInput[0] < dimensionOutput[0]:
        return False, {}
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
    dimensionExtra = self.addLayer(layerStart, layerEnd, typeLayer, dimensionLayer)
    print ('\n@@@ graph is valid:', self.validateGraph(), '@@@')
    historyAction.append('Add')
    return True, { nameLayerEnd: [dimensionExtra] }

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
          for namePast in self.dictGraphReversed[nameNext]:
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
          # dimension for removal
          dimensionTrimed = 0
          if typeNext == 'Conv2d':
            dimensionTrimed = dimensionThis[1]
          elif typeNext == 'Linear':
            if typeThis == 'Conv2d' or (typeThis == 'Input' and len(outputThis) == 3):
              dimensionTrimed = outputThis[0] * outputThis[1] * outputThis[2]
            elif typeThis == 'Linear' or (typeThis == 'Input' and len(outputThis) == 1):
              dimensionTrimed = dimensionThis[1]
          listDimensionTrimed = [dimensionStart, dimensionStart + dimensionTrimed]
          dictPathTrimed[nameNext].append(listDimensionTrimed)
    # sort and merge trimed path's dimension
    print (dictPathTrimed)
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
    print (dictPathTrimed)

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
    # trim layer information
    self.dictGraph = copy.deepcopy(dictGraph)
    for nameIsolated in listIsolated:
      del self.dictGraph[nameIsolated]
      del self.dictLayer[nameIsolated]
      self.listNameLayer.remove(nameIsolated)
      dictPathTrimed[nameIsolated] = []
    return dictPathTrimed

  def actionRemove(self, positionStart, positionEnd, lengthPath):
    print ()
    print ('#############')
    print ('ACTION REMOVE')
    print ('#############')
    # TODO implement recurrent connection
    if positionStart > positionEnd:
      return False, {}
    nameLayerStart = self.getLayerName(positionStart)
    nameLayerEnd = self.getLayerName(positionEnd)
    if nameLayerStart == nameLayerEnd:
      return False, {}
    if 'input' in nameLayerEnd:
      return False, {}
    # pre-requisite for more than 1 distinct path
    if not (len(self.dictGraph[nameLayerStart]) > 1 and len(self.dictGraphReversed[nameLayerEnd]) > 1):
      return False, {}
    dictPath = self.getPath(nameLayerStart, nameLayerEnd)
    # there are less than 2 distinct paths
    if len(dictPath) < 2:
      return False, {}
    listPath = self.flattenPath(dictPath)
    listPathValid = self.filterPath(listPath)
    if len(listPathValid) == 0:
      return False, {}
    listPathValid.sort(key = len)
    indexPath = min(int(lengthPath * len(listPathValid)), len(listPathValid) - 1)
    pathRemove = listPathValid[indexPath]
    pathRemove = [nameLayerStart] + pathRemove
    dictPathTrimed = self.removePath(pathRemove)
    print ('\n@@@ graph is valid:', self.validateGraph(), '@@@')
    historyAction.append('Remove')
    return True, dictPathTrimed

  def createPytorchScript(self):
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
      scriptInit += '(' + scriptDimension[2:] + ')'
    # inference
    scriptForward = '\n\n  def forward(self'
    for nameInput in self.listNameInput:
      scriptForward += ', ' + nameInput
    scriptForward += '):'
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
      scriptForward += nameLayer + '(' + strInput + '))'
    scriptOutput = ''
    for nameOutput in self.listNameOutput:
      scriptOutput += '\n    ' + nameOutput + ' = output_' + self.dictGraphReversed[nameOutput][0]
    scriptForward += scriptOutput
    # return
    scriptReturn = ''
    for nameOutput in self.listNameOutput:
      scriptReturn += ', ' + nameOutput
    scriptReturn = '\n    return ' + scriptReturn[2:]
    script = scriptLibary + scriptClass + scriptInit + scriptForward + scriptReturn
    return script

input1 = _layer('input1', 'Input', [120, 120, 3])
input2 = _layer('input2', 'Input', [32])
layer1 = _layer('layer1', 'Conv2d', [3, 32, 3, 1, 1])
layer2 = _layer('layer2', 'Conv2d', [32, 64, 4, 4, 0])
layer3 = _layer('layer3', 'Conv2d', [64, 128, 5, 5, 0])
layer4 = _layer('layer4', 'Linear', [4608, 128])
layer5 = _layer('layer5', 'Linear', [128 + 32, 128])
layer6 = _layer('layer6', 'Linear', [128, 64])
output1 = _layer('output1', 'Output', [64])

listNameInput = ['input1', 'input2']
listNameOutput = ['output1']
listLayerName = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']
dictLayer = {'input1': input1, 'input2': input2, 'layer1': layer1, 'layer2': layer2, 'layer3': layer3, 'layer4': layer4, 'layer5': layer5, 'layer6': layer6, 'output1': output1}
dictGraph = {'input1': ['layer1'], 'input2': ['layer5'], 'layer1': ['layer2'], 'layer2': ['layer3'], 'layer3': ['layer4'], 'layer4': ['layer5'], 'layer5': ['layer6'], 'layer6': ['output1']}

mk1 = _net('mk1', listNameInput, listNameOutput, listLayerName, dictLayer, dictGraph)
print ('\n@@@ graph is valid:', mk1.validateGraph(), '@@@')
script = mk1.createPytorchScript()
fileScript = open("net.py", "w")
fileScript.write(script)
fileScript.close()
import net
imp.reload(net)
modelNow = net._net()

countNotEqual = 0
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
  print ('Add:', countAdd, 'Remove:', countRemove, 'Not Equal:', countNotEqual)
  # action
  tensorAction = []
  for counter in range(16):
    tensorAction.append(random.random())
  actionAdd = tensorAction[0]
  actionRemove = tensorAction[1]
  actionRead = tensorAction[2]
  isValidAction = False
  dictDiffLayer = {}
  if actionRead > actionAdd and actionRead > actionRemove:
    continue
  if actionAdd > actionRemove and actionAdd > actionRead:
    positionStartAdd = tensorAction[3]
    positionEndAdd = tensorAction[4]
    sizeFilter = tensorAction[5]
    numFilter = tensorAction[6]
    sizeLinear = tensorAction[7]
    isValidAction, dictDiffLayer = mk1.actionAdd(positionStartAdd, positionEndAdd, sizeFilter, numFilter, sizeLinear)
  elif actionRemove > actionAdd and actionRemove > actionRead:
    positionStartRemove = tensorAction[8]
    positionEndRemove = tensorAction[9]
    lengthPathRemove = tensorAction[10]
    isValidAction, dictDiffLayer = mk1.actionRemove(positionStartRemove, positionEndRemove, lengthPathRemove)
  if isValidAction:
    modelOld = modelNow
    script = mk1.createPytorchScript()
    fileScript = open("net.py", "w")
    fileScript.write(script)
    fileScript.close()
    import net
    imp.reload(net)
    modelNow = net._net()
    if actionAdd > actionRemove and actionAdd > actionRead:
      stateDict = modelNow.state_dict()
      for key in modelOld.state_dict():
        nameLayer = key.split('.')[0]
        if 'weight' in key and nameLayer in dictDiffLayer:
          paraOld = modelOld.state_dict()[key].cpu().data.numpy()
          paraNow = modelNow.state_dict()[key].cpu().data.numpy()
          # TODO: initialize with small values instead of zeros
          paraNow = np.zeros(paraNow.shape)
          typeLayer = mk1.dictLayer[nameLayer]._type
          if typeLayer == 'Conv2d':
            paraNow[:, : dictDiffLayer[nameLayer][0][0], :, :] = paraOld
          elif typeLayer == 'Linear':
            paraNow[:, : dictDiffLayer[nameLayer][0][0]] = paraOld
          stateDict[key] = torch.tensor(paraNow).cuda()
        else:
          stateDict[key] = modelOld.state_dict()[key]
      modelNow.load_state_dict(stateDict)
    elif actionRemove > actionAdd and actionRemove > actionRead:
      print ('\n')
      print (dictDiffLayer)
      stateDictNow = modelNow.state_dict()
      stateDictOld = modelOld.state_dict()
      for key in modelOld.state_dict():
        nameLayer = key.split('.')[0]
        if nameLayer in dictDiffLayer:
          if len(dictDiffLayer[nameLayer]) == 0:
            paraOld = modelOld.state_dict()[key].cpu().data.numpy()
            paraZeros = np.zeros(paraOld.shape)
            stateDictOld[key] = torch.tensor(paraZeros).cuda()
          elif 'weight' in key and len(dictDiffLayer[nameLayer]) > 0:
            dimensionTrimed = dictDiffLayer[nameLayer]
            typeLayer = mk1.dictLayer[nameLayer]._type
            paraOld = modelOld.state_dict()[key].cpu().data.numpy()
            paraNow = modelNow.state_dict()[key].cpu().data.numpy()
            for pathTrimed in dimensionTrimed:
              # TODO: gradually replace parameters with zeros
              paraZeros = np.zeros(paraOld.shape)
              if typeLayer == 'Conv2d':
                paraOld[:, pathTrimed[0] : pathTrimed[1], :, :] = paraZeros[:, pathTrimed[0] : pathTrimed[1], :, :]
              elif typeLayer == 'Linear':
                paraOld[:, pathTrimed[0] : pathTrimed[1]] = paraZeros[:, pathTrimed[0] : pathTrimed[1]]
              stateDictOld[key] = torch.tensor(paraOld).cuda()
            print ('\n@@@@@@@@@@@@@@@\nlayer:', nameLayer)
            print ('\ndimension Trimed:', dimensionTrimed)
            print ('\nold dimension:', paraOld.shape)
            print ('now dimension:', paraNow.shape)
            startOld = 0
            endOld = 0
            startNow = 0
            endNow = 0
            for interval in range(len(dimensionTrimed) + 1):
              print ('\ninterval:', interval)
              if interval == 0:
                if dimensionTrimed[0][0] != 0:
                  startOld = 0
                  endOld = dimensionTrimed[0][0]
                  startNow = 0
                  endNow = dimensionTrimed[0][0]
                  print ('a')
                  print ('old:', 0, dimensionTrimed[0][0])
                  print ('now:', 0, dimensionTrimed[0][0])
                else:
                  continue
              elif interval == len(dimensionTrimed):
                if dimensionTrimed[interval - 1][1] != paraOld.shape[1]:
                  startOld = dimensionTrimed[len(dimensionTrimed) - 1][1]
                  endOld = paraOld.shape[1]
                  startNow = endNow
                  endNow = paraNow.shape[1]
                  print ('b')
                  print ('old:', dimensionTrimed[len(dimensionTrimed) - 1][1], paraOld.shape[1])
                  print ('now:', startNow, paraNow.shape[1])
                else:
                  continue
              else:
                if True:#dimensionTrimed[interval - 1][1] != dimensionTrimed[interval][0]:
                  startOld = dimensionTrimed[interval - 1][1]
                  endOld = dimensionTrimed[interval][0]
                  startNow = endNow
                  endNow = startNow + dimensionTrimed[interval][0] - dimensionTrimed[interval - 1][1]
                  print ('c')
                  print ('old:', dimensionTrimed[interval - 1][1], dimensionTrimed[interval][0])
                  print ('now:', startNow, startNow + dimensionTrimed[interval][0] - dimensionTrimed[interval - 1][1])
                else:
                  continue
              if typeLayer == 'Conv2d':
                paraNow[:, startNow : endNow, :, :] = paraOld[:, startOld : endOld, :, :]
              elif typeLayer == 'Linear':
                paraNow[:, startNow : endNow] = paraOld[:, startOld : endOld]
              stateDictNow[key] = torch.tensor(paraNow).cuda()
        else:
          stateDictNow[key] = modelOld.state_dict()[key]
      modelOld.load_state_dict(stateDictOld)
      modelNow.load_state_dict(stateDictNow)
    # testing output for both models
    modelOld.cuda()
    modelNow.cuda()
    input1 = torch.rand(1, 3, 120, 120).cuda()
    input2 = torch.rand(1, 32).cuda()
    outputOld = modelOld(input1, input2).cpu().data.numpy()
    outputNew = modelNow(input1, input2).cpu().data.numpy()
    if (outputOld == outputNew).all():
      print ('\n@@@ models\' outputs are equivalent @@@')
    else:
      print ('\n@@@ models\' outputs are not equal @@@')
      #if actionRemove > actionAdd:
      if True:
        countNotEqual += 1
        print (outputOld)
        print (outputNew)
        counter = 0
        for count in range(len(outputOld[0])):
          if outputOld[0][count] != outputNew[0][count]:
            counter += 1
            print (count, outputOld[0][count] - outputNew[0][count])
        print (counter)
    # after training (fill all zeros with small values)
    stateDictNow = modelNow.state_dict()
    for key in modelNow.state_dict():
      nameLayer = key.split('.')[0]
      if 'weight' in key and nameLayer in dictDiffLayer:
        paraNow = modelNow.state_dict()[key].cpu().data.numpy()
        bound = 1.0
        if mk1.dictLayer[nameLayer]._type == 'Conv2d':
          bound = 1.0 / ((paraNow.shape[1] * paraNow.shape[2] * paraNow.shape[3]) ** 0.5)
        elif mk1.dictLayer[nameLayer]._type == 'Linear':
          bound = 1.0 / (paraNow.shape[1] ** 0.5)
        paraInit = np.random.uniform(-bound, bound, paraNow.shape)
        for pathInit in dictDiffLayer[nameLayer]:
          if mk1.dictLayer[nameLayer]._type == 'Conv2d':
            paraNow[:, pathInit[0] : pathInit[1], :, :] = paraInit[:, pathInit[0] : pathInit[1], :, :]
          elif mk1.dictLayer[nameLayer]._type == 'Linear':
            paraNow[:, pathInit[0] : pathInit[1]] = paraInit[:, pathInit[0] : pathInit[1]]
        stateDictNow[key] = torch.tensor(paraNow).cuda()
      modelNow.load_state_dict(stateDictNow)
