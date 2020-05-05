#!/usr/bin/env ruby

require 'octokit'
require 'securerandom'
require "open-uri"
require "fileutils"

def download(url, path)
  case io = URI.open(url)
  when StringIO then File.open(path, 'w') { |f| f.write(io) }
  when Tempfile then io.close; FileUtils.mv(io.path, path)
  end
end

nFollowers = ARGV[0] # 1000 == About 2K users currently
if nFollowers.nil?
  raise "Parameter <number of followers> not specified"
end
filterLanguage = ARGV[1]
if filterLanguage.nil?
  raise "Parameter <filter language> not specified"  # application/x-ruby for ruby
end
waitSeconds = ARGV[2]
waitSeconds ||= 2

# Create data dir if not exists
dataDir = "./data"
Dir.mkdir(dataDir) unless Dir.exist?(dataDir)

# Create Oktokit client to acces Github API
client = Octokit::Client.new(:access_token => "8d06b8f677f17cacd331530745d2eca53ff7609c")
client.auto_paginate = true

# Search users with many followers
usersFound = client.search_users("followers:>#{nFollowers}")
usersId = usersFound.items.map { |u| u.login }
puts("#{usersId.count} users found with #{nFollowers} followers")

usersId.each do |userId|
  puts(userId)
  sleep waitSeconds.to_f
  gistsFound = client.gists(userId)
  puts("#{gistsFound.count} number of Gists found.")
  gistsFound.each do |gist|
    # gist.files.each {|f| puts f[1].class}
    gist.files.each do |file|
      if file[1].language == filterLanguage
        sleep waitSeconds.to_f
        outFileName = "#{dataDir}/#{SecureRandom.uuid}_#{file[1].filename}"
        puts(outFileName)
        download(file[1].raw_url, outFileName)
      end
    end
  end
end

