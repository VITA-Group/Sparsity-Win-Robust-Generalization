from models.preactivate_resnet import *

# scale  0.223
# planes_list:  [14, 29, 57, 114]
# small dense density 4.98%

# scale  0.316
# planes_list:  [20, 40, 81, 162]
# small dense density 10.01%   

# scale  0.448
# planes_list:  [29, 57, 115, 229]
# small dense density 20.06%

# scale  0.775
# planes_list:  [50, 99, 198, 397]
# small dense density 60.06%

# scale  0.895
# planes_list:  [57, 115, 229, 458]
# small dense density 80.05%

def densityToScale(density):
    d = {0.05: 0.223, 
         0.1: 0.316, 
         0.2: 0.448, 
         0.6: 0.775,
         0.8: 0.895}
    
    return d[density]

def getSmallDenseResNet18(density, classes=10):
    return SmallDenseResNet18(densityToScale(density), num_classes = classes)