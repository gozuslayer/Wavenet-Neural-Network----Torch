require 'optim'
require 'socket'
require 'pl'
require 'csvigo'
require 'audio'

local utils = require 'utils.lua_utils'

------------------------------------------------------------------------------------------------
------------------------------------- CLASS NETWORK --------------------------------------------
------------------------------------------------------------------------------------------------
local network = {}


function network:init(opt)

    -- Initialize network
    self.tnt = require 'torchnet'

    --setting up GPU
    --self:setup_gpu(opt)

    -- creating saving directory
    self:create_saving_directory(opt)   

    -- saving parameters of model
    self:saving_parameters_of_model(opt)

    -- follow stats and plot loss of training and validation
    self:setup_logger(opt)

    self:setup_model(opt)
    self:setup_train_engine(opt)
end


------------------------------------------------------------------------------------------------
----------------------------------------- GPU ---------------------------------------------
------------------------------------------------------------------------------------------------


function network:setup_gpu(opt)
    if opt.gpuids ~= '' then
        require 'cunn'
        require 'cutorch'
        -- require 'cudnn'
        if string.len(opt.gpuids) == 1 then
            cutorch.setDevice(self:map_gpuid(tonumber(opt.gpuids)))
            cutorch.manualSeed(123)
        end
        print(string.format('Using GPUs %s', opt.gpuids))
    end
end
------------------------------------------------------------------------------------------------
----------------------------------------- TRAINING ---------------------------------------------
------------------------------------------------------------------------------------------------
function network:setup_model(opt)

    --setting Gradient descent optimizer from optim
    local method = {
        sgd = optim.sgd,
        adam = optim.adam,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        rmsprop = optim.rmsprop,
        adamax = optim.adamax
    }
    self.optim_method = method[opt.method]

    

    -- Initiating model with architecture p
    local model = require 'model'
    model.init(opt)

    --getting the no conditional
    self.nets = model.get_nocond_net()

    --getting criterion
    self.criterions = model.get_criterion()

    -- getting iterator for training and valid 
    -- train iterator
    self.train_iterator = self:get_iterator('train', opt)

    -- validation iterator (to evaluate model)
    self.valid_iterator = self:get_iterator('valid', opt)
end

function network:setup_train_engine(opt)
    -- Setting up engine for training and validation
    -- and defining hooks for training 
    --train engine
    self.engine = self.tnt.OptimEngine()
    self.train_meter  = self.tnt.AverageValueMeter()

    self.valid_engine = self.tnt.SGDEngine()
    self.valid_meter = self.tnt.AverageValueMeter()
    
    -- setting timer time of one epoch
    self.timer = self.tnt.TimeMeter{unit=true}

    -------------------------------
    -- Hooks for training engine --
    -------------------------------
    self.engine.hooks.onStartEpoch = function(state)

        --reset evaluation meter
        self.train_meter:reset()
    end


    self.engine.hooks.onForwardCriterion = function(state)

        -- add value of each sample
        self.train_meter:add(state.criterion.output)

        -- print avg loss after sample
        print(string.format('Epoch: %d; Sample: %d; avg. loss: %2.4f',state.epoch,state.t, self.train_meter:value()))
    end


    self.engine.hooks.onEndEpoch = function(state)

        -- Getting loss on validation to detect overfitting
        local validation_loss = math.huge
        if state.epoch % opt.eval_model_every_epoch == 0 then 
            print('Getting validation loss')
            self.valid_engine:test{
                network   = self.nets,
                iterator  = self.valid_iterator,
                criterion = self.criterions
            }

            -- see hooks for validation engine for complete 
            validation_loss = self.valid_meter:value()
        end

        local train_loss = self.train_meter:value()
        self.logger:add{train_loss, valid_loss, self.timer:value()}
        

        -- Timer
        self.timer:incUnit()
        print(string.format('Time for one epoch : %.4f',self.timer:value()))

        -- Save model and loss
        if (state.epoch % opt.save_model_every_epoch == 0) then
            local fn = string.format('wavenet_%d.t7', state.epoch)
            self:save_network(fn)
        end
    end

    -- Saving network on End 
    self.engine.hooks.onEnd = function(state)
        local fn = string.format('net_e%d.t7', state.epoch)
        self:save_network(fn)
    end

    -- Hooks for validation engine
    -- getting validation loss to detect overfitting
    self.valid_engine.hooks.onStartEpoch = function(state)
        self.valid_meter:reset()
    end
    self.valid_engine.hooks.onForwardCriterion = function(state)
        self.valid_meter:add(state.criterion.output)
    end
    self.valid_engine.hooks.onEnd = function(state)
        print(string.format('Validation avg. loss: %2.4f',self.valid_meter:value()))
    end
end

function network:create_saving_directory(opt)
    -- Create directory to save models
    local cur_dt = os.date('*t', socket.gettime())
    local name = string.format('%d%d%d__%dH%dM%dS',cur_dt.year, cur_dt.month, cur_dt.day, cur_dt.hour, cur_dt.min, cur_dt.sec)
    saving_path = path.join(opt.models_dir, name)
    utils.make_dir_if_not_exists(saving_path)
    self.save_path = saving_path
end

function network:saving_parameters_of_model(opt)
    local fp = path.join(self.save_path, 'cmd')
    torch.save(fp .. '.t7', opt)
    csvigo.save{path=fp .. '.csv', data=utils.convert_table_for_csvigo(opt)}
end

function network:save_network(fn)
    local fp = path.join(self.save_path, fn)
    print(string.format('Saving model to: %s', fp))
    torch.save(fp, self.nets)
end


function network:setup_logger(opt)
    local fp = path.join(self.save_path, 'stats.log')
    self.logger = optim.Logger(fp)
    self.logger:setNames{'Train loss', 'Valid loss', 'Avg. epoch time'}
end

function network:train(opt)
    self.engine:train{
        network   = self.nets,
        iterator  = self.train_iterator,
        criterion = self.criterions,
        optimMethod = self.optim_method,
        config = {
            learningRate = opt.lr,
            learningRateDecay = opt.lr_decay,
            momentum = opt.mom,
            dampening = opt.damp,
            nesterov = opt.nesterov,
        },
        maxepoch  = opt.maxepochs,
    }
end

function network:get_iterator(split, opt)
    return self.tnt.ParallelDatasetIterator{
        nthread = 1,
        init = function() require 'torchnet' end,
        closure = function()
            local tnt = require 'torchnet'
            require 'dataset'
            local dataset
            dataset = tnt.NocondDataset(split, opt)
            return dataset
        end,
    }
end

------------------------------------------------------------------------------------------------
----------------------------------------- TESTING ----------------------------------------------
------------------------------------------------------------------------------------------------

function network:generate(opt)
    -- Load model and parameters
    require 'model'
    -- path of model and parameters
    local model_path = path.join(opt.models_dir, opt.load_model_dir, opt.load_model_name)
    local cmd_path = path.join(opt.models_dir, opt.load_model_dir, 'cmd.csv')

    -- load parameters and model
    local param_of_training = utils.read_cmd_csv(cmd_path)
    local net = torch.load(model_path)
    print('model loaded')

    -- Create initial input with 0 of size 1,1,1,rf
    local x = torch.Tensor(1, 1, 1, traintime_opt.receptive_field_size):zero()

    -- Create output sequentially : when an output is predict reinject at end of x 
    -- Initialize output
    outputs = {}

    local num_samples_to_generate = opt.gen_length

    -- See timer for generation
    local start_time = os.clock()
    local wavenet_utils = require 'utils.wavenet_utils'

    for i=1,num_samples_to_generate  do
        if i % 100 == 0 then
            print(string.format('generating sample %d',i))
        end
        local activations = net:forward(x) -- log softmax 
        local _, bin = torch.max(activations[1][opt.receptive_field_size], 1)
        bin = bin[1]

        -- Decode through inverse mu-law
        local output_val = wavenet_utils.decode(torch.Tensor({bin}), traintime_opt.mu, traintime_opt.max_val_for_quant)[1]

        -- Create next input by shifting and appending output
        x[{{1},{1},{1},{1, traintime_opt.receptive_field_size - 1}}] = x[{{1},{1},{1},{2, traintime_opt.receptive_field_size}}]
        x[1][1][1][traintime_opt.receptive_field_size] = output_val
        table.insert(outputs, output_val)
    end

    local end_time = os.clock()
    print(string.format('%.2f minutes', (end_time - start_time) / 60))
    outputs = torch.Tensor(outputs)     -- (len,)
    outputs = outputs:reshape(outputs:size(1),1)  -- (len,1)


    --saving audio file
    audio.save(opt.gen_name, outputs, 16000)

end

return network
