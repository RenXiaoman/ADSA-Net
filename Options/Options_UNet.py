from Options.BaseOptions import BaseOptions


class Options_UNet(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=20, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_UXNET_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=16, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4)
        self.isTrain = True
        return parser