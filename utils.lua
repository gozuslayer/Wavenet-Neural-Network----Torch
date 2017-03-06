require 'lfs'
require 'pl'

----------------------------------------------------------
------------------ mu-law & quantize --------------------
----------------------------------------------------------
function mu_law_and_quantize(wav, mu, max_value_for_quantize)
    -- mu-law companding transformation
    -- wav in [-1,1]
    wav = wav / max_value_for_quantize
 
    wav = torch.cmul(torch.sign(wav), torch.log(1 + mu * torch.abs(wav)) / torch.log(1 + mu))

    -- quantize values in bins from 1 to mu+1
    wav:apply(function (val)    -- get_bin(val)
        local num_bins = mu + 1
        local bin_width = 2 / (num_bins)
        if val == 1.0 then
            return num_bins
        else
            return math.floor((val + 1) / bin_width) + 1
        end
    end)

    return wav
end

----------------------------------------------------------
------------------ decode from quantization --------------
----------------------------------------------------------
function decode_from_quantization(wav, mu, max_value_for_quantize)
    -- Convert from bin number to value in [-1,1]
    wav:apply(function (bin)
        local num_bins = mu + 1
        local bin_width = 2 / (num_bins)
        local val = (-1 + bin_width / 2) + (bin - 1) * bin_width
        return val
    end)
    -- mu-law expansion
    wav = torch.cmul(torch.sign(wav), (1/mu) * (torch.pow((1+mu), torch.abs(wav)) - 1))
    -- scale
    wav = wav * max_value_for_quantize

    return wav
end

----------------------------------------------------------
---------- Create Directory if doesnt exist --------------
----------------------------------------------------------

function make_directory(dir_path)
	if not path.exists(dir_path) then
		lfs.mkdir(dir_path)
	end
end

----------------------------------------------------------
---------- Reading Line from file ------------------------
----------------------------------------------------------

function read_lines(file_path)
	if not path.exists(file_path) then return {} end
    lines = {}
    for line in io.lines(file_path) do 
        lines[#lines + 1] = line
    end
    return lines
end
