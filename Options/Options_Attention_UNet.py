from Options.BaseOptions import BaseOptions


class Options_A_UNet(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=10, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_Attention_UNet_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=4, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        self.isTrain = True
        return parser
    
class Options_A_UNet_158(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=10, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/Prostate158/nnUNet_train', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_Attention_UNet_158', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1', lr=5e-5) 
        self.isTrain = True
        return parser