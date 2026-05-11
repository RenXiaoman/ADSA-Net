from Options.BaseOptions import BaseOptions


class Options_x_PICAI(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=10, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        self.isTrain = True
        return parser
    
    
class Options_x_PICAI_NewFusion(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_NewFusion', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=16, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_PICAI_New_CNN_Encoder(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_DTWC', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_PICAI_Improved(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Improved', help='the current task name')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
        parser.set_defaults(gpu_ids='1', epoch=1000, lr=3e-4)
        self.isTrain = True
        return parser
    
class Options_x_PICAI_Backbone(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Backbone', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser

class Options_x_PICAI_Backbone_MRE(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Backbone_MRE', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser

class Options_x_PICAI_Backbone_ACF(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Backbone_ACF', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser

class Options_x_PICAI_Backbone_SAEB(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Backbone_SAEB', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1')  # specify GPU ids
        self.isTrain = True
        return parser


class Options_x_PICAI_Backbone_MRE_ACF(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_Backbone_MRE_ACF', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_PICAI_DWIAsLeading(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_DWIAsLeading', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_PICAI_ADCAsLeading(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        # parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')

        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI_ADCAsLeading', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=5, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_158(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/Prostate158/nnUNet_train', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4, gpu_ids='0')
        self.isTrain = True
        return parser