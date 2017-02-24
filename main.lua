-- Main file to train network
require 'utils.lua_utils'

---------------------------------------------------------------------------
-- Params
---------------------------------------------------------------------------
cmd = torch.CmdLine()

cmd:text('options:')
-- Training / Testing
cmd:option('-mode', 'train', 'train or test (generate)')
cmd:option('-dataset', 'vctk', 'vctk or')
cmd:option('-type', 'nocond', 'nocond / text(not implemented) / textplusspeaker(not implemented)')

-- preprocessing / postprocessing : mu-law
cmd:option('-downsample_factor', 3, '48000 samples per second -> 16000 samples per second')
cmd:option('-mu', 255, 'quantization')

-- Generate from model
cmd:option('-load_model_dir', '', 'directory name from which to load model')
cmd:option('-load_model_name', '', 'e.g. net_e3.t7')
cmd:option('-gen_length', 100, 'number of samples to generate')
cmd:option('-gen_name','generated.wav', 'name of sample generated')

-- Architecture of network 
cmd:option('-expblocks', 4, 'number of exponentially dilated blocks')
cmd:option('-max_dilation', 512 , 'max dilation in one block , e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512')
cmd:option('-dilated_channels', 32, 'number of feature maps for dilated convs')
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

-- Saving Model
cmd:option('-split_dirname', 'split_main', 'name of directory containing splits')
cmd:option('-save_model_every_epoch', 10, 'how often to save model')
cmd:option('-eval_model_every_epoch', 1, 'how often to eval model on validation set')

local parameters = cmd:parse(arg)

-- Calculate some things
parameters.models_dir = path.join('models', parameters.dataset, parameters.experiment)
parameters.save_test_dir = path.join('outputs', parameters.dataset, parameters.experiment)

parameters.receptive_field_size = 2*parameters.max_dilation + (parameters.expblocks-1) * (2*parameters.max_dilation - 1)   -- this is also the input size

-- Value calculated by going through wavs and getting max value. This is used to quantize the input with mu law
parameters.max_val_for_quant = 2147418112

---------------------------------------------------------------------------
-- Training
---------------------------------------------------------------------------
local network = require 'network'

if parameters.mode == 'train' then
	network:init(parameters)
    network:train(parameters)
elseif parameters.mode == 'test' then
	network:generate(parameters)
end
