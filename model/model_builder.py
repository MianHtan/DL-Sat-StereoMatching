from model.PSMNet.PSMNet import PSMNet
from model.PSMNet.PSMNet_EdgeRefinement import PSMNet_Edge
from model.GwcNet.PSM_Gwc import PSMNet_Gwc
from model.GwcNet.GwcNet import GwcNet
from model.StereoNet.StereoNet import StereoNet
from model.GCNet.GCNet import GCNet

def build_model(args):
    if args.model == 'GCNet':
        model = GCNet(args.image_channels)
    if args.model == 'PSMNet':
        model = PSMNet(args.image_channels)
    if args.model == 'PSMNet_Edge': 
        model = PSMNet_Edge(args.image_channels)
    if args.model == 'PSMNet_Gwc':
        model = PSMNet_Gwc(args.image_channels, args.groups)
    if args.model == 'GwcNet':
        model = GwcNet(args.image_channels, args.groups)
    if args.model == 'StereoNet':
        model = StereoNet(args.image_channels, args.k, args.refinement_time)

    return model