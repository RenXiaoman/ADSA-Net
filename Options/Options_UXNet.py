from Options.BaseOptions import BaseOptions


class Options_UXNet_PICAI(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=10, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_UXNet_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1', lr=8e-4, epoch=1000)  # specify GPU ids
        self.isTrain = True
        return parser


class Options_UXNet_Chengda(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_UXNet_PICAI', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1', lr=5e-4, epoch=1000)  # specify GPU ids
        self.isTrain = True
        return parser
    
    
class Options_x_NewFusion(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_NewFusion', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=16, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1')  # specify GPU ids
        self.isTrain = True
        return parser

class Options_x_chengda_optimized(BaseOptions):
    """Optimized training options for private dataset with anti-overfitting measures.

    优化说明:
    1. 学习率调整: 1e-3 → 1e-4 (降低10倍，172M大模型需要更稳定训练)
    2. 正则化增强: weight_decay=3e-5 → 1e-4 (增强3倍，防止过拟合)
    3. 批次大小调整: 16 → 8 (减半，提供更多梯度噪声，改善泛化)
    4. 训练轮数控制: 1000 → 500 (减半，防止过拟合)
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')


        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_Optimized', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size (reduced for better generalization)')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')

        # Anti-overfitting parameters - override BaseOptions defaults
        parser.add_argument('--lr', type=float, default=1e-4, help='reduced learning rate for large model')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='increased weight decay for regularization')
        parser.add_argument('--epoch', type=int, default=500, help='reduced epochs to prevent overfitting')

        parser.set_defaults(gpu_ids='1')  # specify GPU ids
        self.isTrain = True
        return parser
    
class Options_x_chengda_New_CNN_Encoder(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_UXNet_New_CNN_Encoder', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=10, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='1', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 

class Options_x_chengda_optimized(BaseOptions):
    """Optimized training options for private dataset with anti-overfitting measures.

    优化说明:
    1. 学习率调整: 1e-3 → 1e-4 (降低10倍，172M大模型需要更稳定训练)
    2. 正则化增强: weight_decay=3e-5 → 1e-4 (增强3倍，防止过拟合)
    3. 批次大小调整: 16 → 8 (减半，提供更多梯度噪声，改善泛化)
    4. 训练轮数控制: 1000 → 500 (减半，防止过拟合)
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')


        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_Optimized', help='the current task name')
        # parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        # parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=8, help='input train batch size (reduced for better generalization)')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')

        # Anti-overfitting parameters - override BaseOptions defaults
        parser.set_defaults(lr=5e-5, weight_decay=1e-4, epoch=2000)
        # parser.set_defaults('--weight_decay', type=float, default=1e-4, help='increased weight decay for regularization')

        parser.set_defaults(gpu_ids='1')  # specify GPU ids
        self.isTrain = True
        return parser