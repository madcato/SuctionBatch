#!/usr/bin/env ruby

require 'octokit'

# Create data dir if not exists
dataDir = "./data"
Dir.mkdir(dataDir, 0660) unless Dir.exist?(dataDir)

# Create Oktokit client to acces Github API
client = Octokit::Client.new(:access_token => "8d06b8f677f17cacd331530745d2eca53ff7609c")
client.auto_paginate = true

# Search users with many followers
nFollowers = ARGV[0] # 1000 == About 2K users currently
if nFollowers.nil?
  raise "Parameter <number of followers> not specified"
end
# usersFound = client.search_users("followers:>#{nFollowers}")
# usersId = usersFound.items.map { |u| u.login }
# puts("#{usersId.count} users found with #{nFollowers} followers")

# try = usersId[0]
try = "torvalds"
gistsFound = client.gists(try)
p gistsFound

