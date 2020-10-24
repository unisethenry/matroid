import copy
import random

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
    print ('\n@@@ all connection are validation @@@')
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
    numberLayer = len(self.listNameLayer)
    indexLayer = min(int(positionLayer * numberLayer), numberLayer - 1)
    return self.listNameLayer[indexLayer]

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

  def actionAdd(self, positionStart, positionEnd, sizeFilter, numFilter, sizeLinear):
    print ()
    print ('##########')
    print ('ACTION ADD')
    print ('##########')
    # TODO implement recurrent connection
    if positionStart > positionEnd:
      return False
    # TODO include 'input' layer
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
    if layerStart._type == 'Linear' and layerEnd._type == 'Conv2d':
      return False
    if layerStart._type == 'Conv2d' and layerEnd._type == 'Conv2d':
      if dimensionInput[0] < dimensionOutput[0]:
        return False
      print ('################')
      print ('Conv2d to Conv2d')
      print ('################')
      typeLayer = 'Conv2d'
      dimensionLayer = self.getLayerDimensionC2C(dimensionInput, dimensionOutput, numFilter, sizeFilter)
    elif layerStart._type == 'Conv2d' and layerEnd._type == 'Linear':
      print ('################')
      print ('Conv2d to Linear')
      print ('################')
      typeLayer = 'Conv2d'
      dimensionLayer = self.getLayerDimensionC2L(dimensionInput, dimensionOutput, sizeLinear)
    elif layerStart._type == 'Linear' and layerEnd._type == 'Linear':
      print ('################')
      print ('Linear to Linear')
      print ('################')
      typeLayer = 'Linear'
      dimensionLayer = self.getLayerDimensionL2L(dimensionInput, dimensionOutput, sizeLinear)
    self.addLayer(layerStart, layerEnd, typeLayer, dimensionLayer)
    historyAction.append('Add')
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
    for nameInput in listNameInput:
      listNext = self.dictGraph[nameInput]
      while len(listNext) == 1:
        nameNext = listNext[0]
        if 'output' in nameNext:
          break
        listNameLayer.append(nameNext)
        listNext = self.dictGraph[nameNext]
    for nameOutput in listNameOutput:
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
      # early continue such that all isolated layers are found from dictGraph
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
    # TODO compare dictPath with dictGraph to get stacking positions and dimensions of layers
    dictReversed = self.computeReversedGraphAlone(dictGraph)
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
          if typeThis == 'Conv2d':
            dimensionNext[0] -= outputThis[0] * outputThis[1] * outputThis[2]
          elif typeThis == 'Linear':
            dimensionNext[0] -= dimensionThis[1]
    # trim layer information
    self.dictGraph = copy.deepcopy(dictGraph)
    for nameIsolated in listIsolated:
      del self.dictGraph[nameIsolated]
      del self.dictLayer[nameIsolated]
      self.listNameLayer.remove(nameIsolated)
    print ('path removal is valid:', self.validateGraph())
    return True

  def actionRemove(self, positionStart, positionEnd, lengthPath):
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
    listPathValid = sorted(listPathValid, key = len)
    indexPath = min(int(lengthPath * len(listPathValid)), len(listPathValid) - 1)
    pathRemove = listPathValid[indexPath]
    pathRemove = [nameLayerStart] + pathRemove
    self.removePath(pathRemove)
    historyAction.append('Remove')
    return True

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

while True:
  countAdd = 0
  countRemove = 0
  for counter in range(len(historyAction)):
    if historyAction[counter] == 'Add':
      countAdd += 1
    elif historyAction[counter] == 'Remove':
      countRemove += 1
  print ('\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
  print ('Add:', countAdd, 'Remove:', countRemove)
  # action
  tensorAction = []
  for counter in range(16):
    tensorAction.append(random.random())

  actionAdd = tensorAction[0]
  actionRemove = tensorAction[1]
  actionRead = tensorAction[2]

  if actionAdd > actionRemove and actionAdd > actionRead:
    positionStartAdd = tensorAction[3]
    positionEndAdd = tensorAction[4]
    sizeFilter = tensorAction[5]
    numFilter = tensorAction[6]
    sizeLinear = tensorAction[7]
    mk1.actionAdd(positionStartAdd, positionEndAdd, sizeFilter, numFilter, sizeLinear)
    print ('\n@@@ graph is valid:', mk1.validateGraph(), '@@@')
  elif actionRemove > actionAdd and actionRemove > actionRead:
    positionStartRemove = tensorAction[8]
    positionEndRemove = tensorAction[9]
    lengthPathRemove = tensorAction[10]
    mk1.actionRemove(positionStartRemove, positionEndRemove, lengthPathRemove)
  elif actionRead > actionAdd and actionRead > actionRemove:
    continue
