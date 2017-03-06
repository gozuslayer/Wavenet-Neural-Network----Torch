-----------------------------------------------------
-------------------- Parameters --------------------
-----------------------------------------------------

cmd = torch.CmdLine()

-- Training or testing
cmd:option('-mode','train','train or test')
cmd:option('-dataset','vctk','Wav directory')
cmd:option('-type','withoutcond','Type of training : withoutcond (others not implemented)')

-- post/pre-processing : mu-law transformation
cmd:option('-mu',255,'mu for mu-law transformation')
cmd:option('-downsample_factor',3,'48000 --3--> 16000')

--- TESTING
-- generating (testing) options
cmd:option('-loading_model','','')
cmd:option('-generation_length',100,'samples to generate')
cmd:option('-generation_file_name','generated.wav','name of wav generated with model')

--- TRAINING
-- architecture of Wavenet
cmd:option('-expblocks', 3, 'number of exponentially dilated blocks')
cmd:option('-max_dilation', 512 , 'max dilation in one block , e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512')
cmd:option('-dilated_channels', 16, 'number of feature maps for dilated convs')
cmd:option('-res_channels', 4, 'number of feature maps for 1x1 convolutions in residual block')
cmd:option('-output_channels', 8, 'number of feature maps for 1x1 convs in output')

-- Optimization
cmd:option('-maxepochs', 500, 'max number of epochs to train for')
cmd:option('-method','sgd', 'which optimization method to use')
cmd:option('-lr', 0.005, 'learning rate')
cmd:option('-lr_decay', 0, 'learning rate decay')
cmd:option('-mom', 0, 'momentum')
cmd:option('-damp', 0, 'dampening')
cmd:option('-nesterov', false, 'Nesterov momentum')

-- GPU CUDA
cmd:option('-gpuids','','set GPU acceleration')

local opt = cmd:parse(arg)

opt.receptive_field_size = 2*opt.max_dilation + (opt.expblocks-1) * (2*opt.max_dilation - 1)   -- this is also the input size

--VCTK
opt.max_value= 2147418112
--OBAMA
--opt.max_value = 2147483648

local network = require 'network'

if opt.mode == 'train' then
	network:init_network(opt)
	network:train(opt)
elseif opt.mode == 'test' then
	network:generate(opt)
end
