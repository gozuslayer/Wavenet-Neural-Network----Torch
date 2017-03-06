--- Preprocessing Data : splitting in train/valid/test ---
--- use twice for get max value for mu-law and splitting ---

require 'pl'
require 'audio'
require 'utils'

local preprocess = {}

local PERC_TRAIN = 0.8
local PERC_VALID = 0.1
local PERC_TEST = 0.1

local PATH = {}
PATH['vctk'] = {}
PATH['vctk']['WAV_PATH'] = 'data/vctk/wav48/'
PATH['vctk']['OUT_PATH'] = 'data/processed/vctk/split'
PATH['obama'] = {}
PATH['obama']['WAV_PATH'] = 'data/obama/'
PATH['obama']['OUT_PATH'] = 'data/processed/obama/split'


function preprocess:write_split_to_file(tbl, fn)
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

function preprocess:split_vctk(perc_tr, perc_va, perc_te, wav_path, out_path)
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
    make_directory(out_path)
    preprocess:write_split_to_file(training_set, path.join(out_path, 'train.txt'))
    preprocess:write_split_to_file(validation_set, path.join(out_path, 'valid.txt'))
    preprocess:write_split_to_file(testing_set, path.join(out_path, 'test.txt'))
end

function preprocess:split_obama(perc_tr, perc_va, perc_te, wav_path, out_path)
    local training_set, validation_set, testing_set = {}, {}, {}

    -- getting wav  
    j = 0       
    local wav_fps = dir.getfiles(wav_path)
    for i, file in ipairs(wav_fps) do
        if path.extension(file) == '.mp3' then
            print(j, file)
            local wav = audio.load(file)
            local num_samples = wav:size(1)
            local sample = {'obama', file, num_samples}

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
    
    -- Saving in file
    make_directory(out_path)
    preprocess:write_split_to_file(training_set, path.join(out_path, 'train.txt'))
    preprocess:write_split_to_file(validation_set, path.join(out_path, 'valid.txt'))
    preprocess:write_split_to_file(testing_set, path.join(out_path, 'test.txt'))
end

function preprocess:split(opt)
    if opt.dataset == 'vctk' then
        preprocess:split_vctk(PERC_TRAIN,PERC_VALID,PERC_TEST,PATH['vctk']['WAV_PATH'] ,PATH['vctk']['OUT_PATH'])
    elseif opt.dataset == 'obama' then
        preprocess:split_obama(PERC_TRAIN,PERC_VALID,PERC_TEST,PATH['obama']['WAV_PATH'] ,PATH['obama']['OUT_PATH'])
    else
        print('Dataset must be vctk or obama')
    end
end

function preprocess:get_max_wav_value(opt)
    --- VCTK ---
    if opt.dataset == 'vctk' then
        local subdirs = dir.getdirectories('data/vctk/wav48')
        local maximum_value = 0
        for _, subdir in ipairs(subdirs) do         
            local wav_files = dir.getfiles(subdir)
            for i, fp in ipairs(wav_files) do
                if path.extension(fp) == '.wav' then
                    local wav = audio.load(fp)
                    local actual_maximum_value = torch.abs(wav):max()
                    if actual_maximum_value > maximum_value then
                        maximum_value = actual_maximum_value
                    end
                end
            end
        end
        print('Max: ' .. maximum_value)
        --- report in main for quantization

    --- OBAMA ---
    elseif opt.dataset == 'obama' then
        local wav_files = dir.getfiles('data/obama')
        local maximum_value = 0
        for i, fp in ipairs(wav_files) do
            if path.extension(fp) == '.mp3'  then
                local wav = audio.load(fp)
                local actual_maximum_value = torch.abs(wav):max()
                if actual_maximum_value > maximum_value then
                    maximum_value = actual_maximum_value
                end
            end
        end
        print('Max: ' .. maximum_value)
        --- report in main for quantization
    end
end


cmd = torch.CmdLine()
cmd:text()
cmd:option('-type', '', 'split or max_value')
cmd:option('-dataset', 'vctk', 'vctk or obama')
local opt = cmd:parse(arg)

if opt.type == 'split' then
    preprocess:split(opt)
elseif opt.type == 'max_value' then
    preprocess:get_max_wav_value(opt)
end
