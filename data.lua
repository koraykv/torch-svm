--[[
	A simple dataset table, this reads binary
	formatted svm data.
]]--

function svm:dataset(fname)
	if not paths.filep(fname) then
		error('File does not exist ' .. fname)
	end

	local data,maxdim = svm.binread(fname,true)
	local nsamples = #data
	local dataset = {}
	function dataset:size() return nsamples end
	function dataset:nfeature() return maxdim end
	function dataset:setbuffersize(bsize) buffersize = bsize end
	function dataset:getbuffersize() return buffersize end

	setmetatable(dataset,{__index = function(self,ind)
		return data[ind]
	end})

	return dataset
end
