-- Functions that are run once to pre-process data

require 'pl'
require 'audio'
require 'csvigo'
local utils = require 'utils.lua_utils'

local preprocess = {}

--------------------------------------------------------
------------------------ Params ------------------------
--------------------------------------------------------
local PARAMS = {}


PARAMS['PERC_TRAIN'] = 0.8
PARAMS['PERC_VALID'] = 0.1
PARAMS['PERC_TEST'] = 0.1
PARAMS['WAV_PATH'] = 'data/vctk/wav48/'
PARAMS['OUT_PATH'] = 'data/processed/vctk/split_main' 


------------------------------------------------------------------------------------------------------------
-- Data splitting into train, valid, text
------------------------------------------------------------------------------------------------------------
function preprocess.sort_split_by_length(split)
    local function sorter(a,b)
        if (a[3] < b[3]) then return true else return false end
    end
    table.sort(split, sorter)
    return split
end

function preprocess.write_split_to_file(tbl, fn)
    local f = io.open(fn, 'w')
    for i=1,#tbl do
        local speaker_id = tbl[i][1]
        local path = tbl[i][2]
        local num_samples = tbl[i][3]
        if i < #tbl then
            f:write(string.format('%s,%s,%s\n', speaker_id, path, num_samples))
        else
            f:write(string.format('%s,%s,%s', speaker_id, path, num_samples))
        end
    end
end

function preprocess.split_helper(perc_tr, perc_va, perc_te, wav_path, out_path)
    local training_set, validation_set, testing_set = {}, {}, {}

    local subdirs = dir.getdirectories(wav_path)
    j = 0
    for _, subdir in ipairs(subdirs) do
        -- getting wav         
        local wav_fps = dir.getfiles(subdir)

        -- getting speakerId
        local speaker_id = path.basename(subdir)
        for i, file in ipairs(wav_fps) do
            if path.extension(file) == '.wav' then
                print(j, file)
                local wav = audio.load(file)
                local num_samples = wav:size(1)
                local sample = {speaker_id, file, num_samples}

                local ratio = i / #wav_fps
                if ratio <= perc_tr then 

                    table.insert(training_set, sample)
                elseif (ratio > perc_tr) and (ratio <= perc_tr + perc_va) then 
                    table.insert(validation_set, sample)
                else 
                    table.insert(testing_set, sample)
                end

                j = j + 1
            end
        end
    end

    -- Saving in file
    utils.make_dir_if_not_exists(out_path)
    tr = preprocess.sort_split_by_length(tr)
    va = preprocess.sort_split_by_length(va)
    te = preprocess.sort_split_by_length(te)
    preprocess.write_split_to_file(tr, path.join(out_path, 'train.txt'))
    preprocess.write_split_to_file(va, path.join(out_path, 'valid.txt'))
    preprocess.write_split_to_file(te, path.join(out_path, 'test.txt'))
end

function preprocess.split(opt)
    if opt.dataset == 'vctk' then
        preprocess.split_helper(
                PARAMS['PERC_TRAIN'],
                PARAMS['PERC_VALID'],
                PARAMS['PERC_TEST'],
                PARAMS['WAV_PATH'],
                PARAMS['OUT_PATH']
                )
    else
        print('Dataset must be vctk')
    end
end

------------------------------------------------------------------------------------------------------------
-- Get max value across all wavs in order to standardize values between [-1,1] for quantization
------------------------------------------------------------------------------------------------------------
function preprocess.get_max_wav_val_vctk()
    local subdirs = dir.getdirectories('data/vctk/wav48')
    local all_vals = {}
    local max = 0
    for _, subdir in ipairs(subdirs) do         -- each subdir is one person
        local wav_fps = dir.getfiles(subdir)
        print(subdir)
        for i, fp in ipairs(wav_fps) do
            if path.extension(fp) == '.wav' then
                local wav = audio.load(fp)
                local max_val = torch.abs(wav):max()
                if max_val > max then
                    max = max_val
                end
                table.insert(all_vals, max_val)
            end
        end
    end
    table.sort(all_vals)

    csvigo.save{path='data/processed/vctk/maxvals.csv', data=utils.convert_table_for_csvigo(all_vals)}
    print('Max: ' .. max)
end

------------------------------------------------------------------------------------------------------------------------
-- Main
------------------------------------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:option('-fn', '', 'split or get_max_wav_val_vctk')
cmd:option('-fn', '', 'split or get_max_wav_val_vctk')
cmd:option('-dataset', 'vctk', 'vcktk or ')
local opt = cmd:parse(arg)

if opt.fn == 'split' then
    preprocess.split(opt)
elseif opt.fn == 'get_max_wav_val_vctk' then
    preprocess.get_max_wav_val_vctk()
end
