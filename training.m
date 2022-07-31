
%program to train an RCNN to detect stop signs
load('rcnnStopSigns', 'layers')   %loading layers of pre-trained RCNN
load stopSignsTable.mat;     %loading image database for training

Igraph = layerGraph(layers);   %Getting layers
Igraph.Layers;    %Displaying layers.

%Define trainning options
options = trainingOptions('sgdm', 'MiniBatchSize', 32, 'initialLearnRate', 1e-6, 'MaxEpochs', 10);

%Trainning RCNN
rcnn = trainRCNNObjectDetector(stopSigns, layers, options, 'NegativeOverlapRange', [0 0.1]);