from Options.BaseOptions import BaseOptions


class Options_x(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='Step1_SegTumor_DIY', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=3, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        self.isTrain = True
        return parser
    
class Options_Ablation(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_ALIEN_chengda', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=3, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
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
    1. 学习率调整: 1e-3 → 1e-4 (降低10倍, 172M大模型需要更稳定训练)
    2. 正则化增强: weight_decay=3e-5 → 1e-4 (增强3倍, 防止过拟合)
    3. 批次大小调整: 16 → 8 (减半，提供更多梯度噪声，改善泛化)
    4. 训练轮数控制: 1000 → 500 (减半，防止过拟合)
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')


        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_MedianFilter', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size (reduced for better generalization)')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')

        # Anti-overfitting parameters - override BaseOptions defaults
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='increased weight decay for regularization')

        parser.set_defaults(gpu_ids='1', lr=1e-4, epoch=1000)  # specify GPU ids
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
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Latest_Fully_Version', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 


class Options_x_chengda_Backbone(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Backbone', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 

class Options_x_chengda_Backbone_MRE(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Backbone_MRE', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 
    
class Options_x_chengda_Backbone_ACF(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Backbone_ACF', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 
    

class Options_x_chengda_Backbone_SAEB(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=12, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Backbone_SAEB', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 
    
class Options_x_chengda_Backbone_MRE_ACF(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=50, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/ChengdaOnlyCSPca', help='path of the data')
        
        
        parser.add_argument('--task_name', type=str, default='SegTumor_DIY_chengda_Backbone_MRE_ACF', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=16, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=None, help='resume training from checkpoint')
        parser.set_defaults(gpu_ids='0', lr=8e-4)  # specify GPU ids
        self.isTrain = True
        return parser 
