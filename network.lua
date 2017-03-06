
require 'utils'
require 'optim'
require 'socket'
----------------------------------------------------------
----------------- CLASS NETWORK --------------------------
---------------------------------------------------------

local network = {}

function network:init_network(opt)
	--- Initialisation du reseau ---

	-- get torchnet for easier and faster training
	self.tnt = require 'torchnet'

	-- creating saving directory for network in order to test it afterwards
	self:create_directory(opt)
	-- saving parameters for hyperparameters tuning
	self:save_parameters_model(opt)
	-- setting up optim logger for plotting loss afterwards
	self:set_logger(opt)

	-- Setting Wavenet network
	self:get_model(opt)

	-- setting train engine of torchnet for training
	self:set_trainEngine(opt)
	self:move_to_gpu(opt)
end

function network:create_directory(opt)
	local current_date = os.date('*t',socket.gettime())
	local name = string.format('%d%d%d_%dH%dM',current_date.year,current_date.month,current_date.day,current_date.hour,current_date.min)
	local saving_path = path.join('model',opt.dataset,name)
	make_directory(saving_path)

	-- store saving for further training
	self.saving_path = saving_path
end

function network:save_parameters_model(opt)
	local file_path = path.join(self.saving_path,'CmdLine')
	torch.save(file_path .. '.t7',opt)
end

function network:set_logger(opt)
	local file_path = path.join(self.saving_path,'stats.log')
	self.logger = optim.Logger(file_path)
end

function network:get_model(opt)
	-- setting Gradient descent optimizer from optim
    local optim_method = {
        sgd = optim.sgd,
        adam = optim.adam,
        adagrad = optim.adagrad,
        adadelta = optim.adadelta,
        rmsprop = optim.rmsprop,
        adamax = optim.adamax
    }
    self.optim_method = optim_method[opt.method]

    --Initialize model with hyperparameters
    local model = require 'wavenet'
    model:init(opt)

    -- getting the network
    self.nets = model:get_network()

    -- set criterion 
    self.criterions = model:get_criterion()

    --getting iterator for torchnet
    -- train iterator
    self.train_iterator = self:get_iterator('train',opt)

    -- valid iterator (for overfitting)
    self.valid_iterator = self:get_iterator('valid',opt)
end

function network:set_trainEngine(opt)
	-- Setting up engine for training and validation
    -- and defining hooks for training 
    --train engine
    self.engine = self.tnt.OptimEngine()
    self.train_meter  = self.tnt.AverageValueMeter()

    self.valid_engine = self.tnt.SGDEngine()
    self.valid_meter = self.tnt.AverageValueMeter()
    self.engines = {self.engine, self.valid_engine}
    
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
        
        print('Getting validation loss')
        self.valid_engine:test{
                network   = self.nets,
                iterator  = self.valid_iterator,
                criterion = self.criterions
            }

        -- see hooks for validation engine for complete 
        validation_loss = self.valid_meter:value()
       

        local train_loss = self.train_meter:value()
        self.logger:add{train_loss, valid_loss, self.timer:value()}
        

        -- Timer
        self.timer:incUnit()
        print(string.format('Time for one epoch : %.4f',self.timer:value()))

        -- Save model EVERY  2 epochs
        if (state.epoch % 2 == 0) then
            local file_name = string.format('wavenet_%d.t7', state.epoch)
            self:save_network(file_name)
        end
    end

    -- Saving network on End 
    self.engine.hooks.onEnd = function(state)
        local file_name = string.format('net_e%d.t7', state.epoch)
        self:save_network(file_name)
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

function network:save_network(fn)
    local fp = path.join(self.saving_path, fn)
    print(string.format('Saving model to: %s', fp))
    torch.save(fp, self.nets)
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
        maxepoch  = opt.maxepochs
    }
end


function network:generate(opt)
    -- Load model and parameters used at train time
    require 'model'
    local model_path = path.join('models', opt.dataset, opt.experiment, opt.load_model_dir, opt.load_model_name)
    local cmd_path = path.join('models', opt.dataset, opt.experiment, opt.load_model_dir, 'cmd.csv')
    local traintime_opt = utils.read_cmd_csv(cmd_path)
    local net = torch.load(model_path)
    print('model loaded')

    -- Create initial input
    local x = torch.Tensor(1, 1, 1, traintime_opt.receptive_field_size):zero()
    if opt.gpuids ~= '' then
        x = x:cuda()
    end

    -- Create output sequentially
    outputs = {}
    local num_samples = opt.gen_length
    local start_time = os.clock()
    local wavenet_utils = require 'utils.wavenet_utils'
    for i=1,num_samples do
        if i % 100 == 0 then
            print(i)
        end
        local activations = net:forward(x)
        local _, bin = torch.max(activations[1][opt.receptive_field_size], 1)
        bin = bin[1]

        print(bin)
        -- Decode through inverse mu-law
        local output_val = decode_from_quantization(torch.Tensor({bin}), traintime_opt.mu, traintime_opt.max_val_for_quant)[1]
        -- Create next input by shifting and appending output
        x[{{1},{1},{1},{1, traintime_opt.receptive_field_size - 1}}] = x[{{1},{1},{1},{2, traintime_opt.receptive_field_size}}]
        x[1][1][1][traintime_opt.receptive_field_size] = output_val
        table.insert(outputs, output_val)
    end
    local end_time = os.clock()
    print(string.format('%.2f minutes', (end_time - start_time) / 60))
    outputs = torch.Tensor(outputs)     -- (len,)
    outputs = outputs:reshape(outputs:size(1),1)  -- (len,1)
    audio.save('nocond.wav', outputs, 16000)

end

function network:get_iterator(split, opt)
    return self.tnt.ParallelDatasetIterator{
        nthread = 1,
        init = function() require 'torchnet' end,
        closure = function()
            local tnt = require 'torchnet'
            require 'dataset'
            local dataset 
            dataset = tnt.WithoutConditionDataset(split, opt)
            return dataset
        end,
    }
end

function network:move_to_gpu(opt)
    if opt.gpuids ~= '' then
        require 'cunn'
        require 'cutorch'
        print('Using GPU')
        self.nets = self.nets:cuda()
        self.criterions = self.criterions:cuda()
        

        local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
        for i,engine in ipairs(self.engines) do
            engine.hooks.onSample = function(state)
                igpu:resize(state.sample.input:size() ):copy(state.sample.input)
                tgpu:resize(state.sample.target:size()):copy(state.sample.target)
                state.sample.input  = igpu
                state.sample.target = tgpu
            end
        end
    end
end


return network









    

