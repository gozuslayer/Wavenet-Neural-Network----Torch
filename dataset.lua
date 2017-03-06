--- Implementation of dataset for torchnet ---

require 'audio'
require 'utils'

local tnt = require 'torchnet'

--- Path to Data ---
local PATH = {}
PATH['vctk'] = 'data/processed/vctk/split/'
PATH['obama'] = 'data/processed/obama/split/'


--- torchnet Dataset ---
local WavenetDataset, _ = torch.class('tnt.WavenetDataset', 'tnt.Dataset', tnt)

function WavenetDataset:__init(split, opt)
	self.mu = opt.mu
	self.max_value = opt.max_value
	self.downsample_factor = opt.downsample_factor
	if opt.dataset == 'vctk' then
		self.Path = PATH['vctk']
	elseif opt.dataset == 'obama' then
		self.Path = PATH['obama']
	end
	self.wav_lines = read_lines( path.join(self.Path, split .. '.txt'))

	self.n = #self.wav_lines
end

function WavenetDataset:size()
	return self.n
end


--- WITHOUT CONDITIONING ---

local WithoutConditionDataset, _ = torch.class('tnt.WithoutConditionDataset', 'tnt.WavenetDataset', tnt)

function WithoutConditionDataset:get(idx)
	local speakerid, wavpath, _ = unpack(utils.split(self.wav_lines[idx], ','))
	-- Load wav and downsample, e.g. from 48000 to 16000
	local wav = audio.load(wavpath)		-- (numsamples_in_wav, 1)
	if wav:size(1) % self.downsample_factor ~= 0 then 	-- pad so we can reshape
		local pad_length = self.downsample_factor - (wav:size(1) % self.downsample_factor)
		wav = torch.cat(wav, torch.zeros(pad_length), 1)
	end
	wav = wav:reshape(wav:size(1) / self.downsample_factor, self.downsample_factor)
	wav = wav[{{},{1}}]		
	
	local input = mu_law_and_quantize(wav, self.mu, self.max_value):transpose(1,2) -- (nsamples,1) -> (1,nsamples)
	local target = input:clone()									-- (1, nsamples)
	input = input:reshape(1,1,input:size(1), input:size(2))			-- Add 1st dimension for batch, 2nd dimension for feature maps
	
	return {
				input = input,
				target = target
			}
end
