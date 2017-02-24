local mnist = require 'mnist'

local dataset = mnist['train' .. 'dataset']()

print (dataset)