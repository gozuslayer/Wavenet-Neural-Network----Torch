require 'nn'
local tnt = require 'torchnet'
local mnist = require 'mnist'

local function getIterator(mode)
	return tnt.ParallelDatasetIterator{
		nthread = 1,
		init = function() require 'torchnet' end,
		closure = function()
			local dataset = mnist[mode .. 'dataset']()
			return tnt.BatchDataset({
				batchsize = 128,
				dataset = tnt.ListDataset{
					list = torch.range(1, dataset.data:size(1)),
					load = function(idx)
						return {
								input = dataset.data[idx],
								target = torch.LongTensor{dataset.label[idx] }
								} -- sample contains input and target
						end
					}	
				})
		end
		}
	end

local net = nn.Sequential():add(nn.Linear(784,10))

local engine = tnt.SGDEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter{topk = {1}}
engine.hooks.onStartEpoch = function(state)
	meter:reset()
	clerr:reset()
end
engine.hooks.onForwardCriterion = function(state)
	meter:add(state.criterion.output)
	clerr:add(state.network.output, state.sample.target)
	print(string.format('avg. loss: %2.4f; avg. error: %2.4f',meter:value(), clerr:value{k = 1}))
end

local criterion = nn.CrossEntropyCriterion()

engine:train{
	network = net,
	iterator = getIterator('train'),
	criterion = criterion,
	lr = 0.1,
	maxepoch = 10,
}
