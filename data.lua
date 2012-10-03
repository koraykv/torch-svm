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


	-- be careful , this is just for experimentation, it will be very very very slooooooow.
	local dense = false
	function dataset:dense()
		dense = true
	end

	local dx
	local function todense(ind,x)
		dx = dx or torch.FloatTensor(maxdim)
		dx:zero()
		for i=1,ind:size(1) do
			dx[ind[i]] = x[i]
		end
		return {nil,dx}
	end

	setmetatable(dataset,{__index = function(self,ind)
		if dense then
			local ex = data[ind]
			return {ex[1],todense(ex[2][1],ex[2][2])}
		else
			return data[ind]
		end
	end})

	return dataset
end
