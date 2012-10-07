--[[
	Implementation of some simple linear classifiers to test with 
	Leon Bottou's implementation for comparing in terms of speed
	and convergence properties.

	hehe, this seems like ~10 times slower
]]--

local svmsgd = torch.class('svm.SvmSgd')

function svmsgd:__init(nf,lam)
	-- weights/biases
	self.w = torch.FloatTensor(nf):zero()
	self.b = 0
	self.wdiv = 1
	self.lambda = lam
	self.eta0 = 1
	self.nupdate = 0
	self.nf = nf
	self.regbias = false
	self.svmloss = svm.hingeloss
end

function svmsgd:loss(a,y)
	return self.svmloss(a,y)
end

function svmsgd:renorm()
	if self.wdiv ~= 1 then
		self.w:mul(1/self.wdiv)
		self.wdiv = 1
	end
end

function svmsgd:wnorm()
	local  w = self.w
	local norm = torch.dot(w,w) / self.wdiv / self.wdiv
	if self.regbias then
		norm = norm + self.b + self.b
	end
	return norm
end

function svmsgd:testOne(y,si,sx)

	-- local variables
	local w    = self.w
	local wdiv = self.wdiv
	local b    = self.b

	local s = svm.spdot(w,si,sx)/wdiv + b

	-- calculate gradient from loss
	local lx,ldx = self:loss(s,y)
	local e = 0
	if s*y <= 0 then e = 1 end
	return s,lx,e
end

function svmsgd:trainOne(y,si,sx,eta)
	local w    = self.w
	local wdiv = self.wdiv
	local b    = self.b
	local lambda = self.lambda

	local s = svm.spdot(w,si,sx)/wdiv + b

	-- update wdiv
	wdiv = wdiv / (1 - eta*lambda)
	if wdiv > 1e5 then
		self:renorm()
		wdiv = self.wdiv
	end

	-- calculate gradient from loss
	local lx,ldx = self:loss(s,y)

	-- update weights
	if ldx ~= 0 then
		svm.spadd(w,eta*ldx*wdiv,si,sx)
	end

	-- update bias
	if self.regbias then
		b = b * (1- eta * 0.01 * lambda)
	end
	b = b + eta*0.01*ldx

	-- put back modified numbers
	self.wdiv = wdiv
	self.b = b
end

function svmsgd:trainepoch(data)
	print('Training on ' .. data:size() .. ' samples')
	-- local variables
	local eta = 0
	local lambda = self.lambda
	local eta0 = self.eta0
	local nupdate = self.nupdate

	for i=1,data:size() do
		-- update learning rate
		eta = eta0 / (1 + lambda*eta0*nupdate)

		-- train for a sample
		local ex = data[i]
		self:trainOne(ex[1], ex[2][1], ex[2][2], eta)

		nupdate = nupdate + 1
	end
	io.write('wNorm=' .. string.format('%.2f',self:wnorm()))
	io.write(' wBias=' .. string.format('%.2f\n',self.b))
	self.nupdate = nupdate
end

function svmsgd:test(data)

	io.write('Testing on ' .. data:size() .. ' samples\n')

	local loss = 0
	local nerr = 0
	for i=1,data:size() do
		local ex = data[i]
		local s,l,e = self:testOne(ex[1], ex[2][1], ex[2][2])
		loss = loss + l
		nerr = nerr + e
	end

	loss = loss/data:size()
	nerr = nerr/data:size()

	io.write('Loss=' .. string.format('%.8f',loss))
	io.write(' Cost=' .. string.format('%.8f',loss + 0.5*self.lambda*self:wnorm()))
	io.write(' Misclassification=' .. string.format('%.2f %%\n',100*nerr))
end

function svmsgd:predict(data)
	local tlabel = torch.IntTensor(data:size())
	local tdec = torch.Tensor(data:size())
	local loss = 0
	local nerr = 0
	for i=1,data:size() do
		local ex = data[i]
		local s,l,e = self:testOne(ex[1], ex[2][1], ex[2][2])
		loss = loss + l
		nerr = nerr + e
		if e == 1 then tlabel[i] = -ex[1] else tlabel[i] = ex[1] end
		tdec[i] = s
	end

	loss = loss/data:size()
	nerr = nerr/data:size()
	io.write('Accuracy=' .. string.format('%.4f %% (%d/%d)\n',
		100-100*nerr,data:size()-nerr*data:size(),data:size()))
	return tlabel,{100-100*nerr,loss,loss + 0.5*self.lambda*self:wnorm()},tdec
end

function svmsgd:train(trdata,tedata,epochs)

	local trtime = torch.Timer()
	for i=1,epochs do
		print('============== Epoch #' .. i .. ' ==============')

		-- train
		trtime:resume()
		self:trainepoch(trdata)
		trtime:stop()
		print('Total Training Time = ' .. string.format('%.2f secs',trtime:time().real))

		-- test
		io.write('>> train: ')
		self:test(trdata)
		if tedata then
			io.write('>> test: ')
			self:test(tedata)
		end
	end
end

function svmsgd:evalEta(nsample,data,eta)
	-- clone the weight and bias
	local w = self.w:clone()
	local b = self.b
	local wdiv = self.wdiv
	for i=1,nsample do
		local ex = data[i]
		self:trainOne(ex[1], ex[2][1], ex[2][2], eta)
	end
	local loss = 0
	for i=1,nsample do
		local ex = data[i]
		local s,l,e = self:testOne(ex[1], ex[2][1], ex[2][2])
		loss = loss + l
	end
	local cost = loss/nsample + 0.5 * self.lambda * self:wnorm()
	self.w:copy(w)
	self.b = b
	self.wdiv = wdiv
	return cost
end

function svmsgd:determineEta0(nsample,data)
	local factor = 2
	local loeta = 1
	local locost = self:evalEta(nsample,data,loeta)
	local hieta = loeta * factor
	local hicost = self:evalEta(nsample,data,hieta)
	if locost < hicost then
		while locost < hicost do
			hieta = loeta
			hicost = locost
			loeta = hieta / factor
			locost = self:evalEta(nsample,data,loeta)
		end
	elseif hicost < locost then
		while hicost < locost do
			loeta = hieta
			locost = hicost
			hieta = loeta * factor
			hicost = self:evalEta(nsample,data,hieta)
		end
	end
	self.eta0 = loeta
	print('# Using eta0='..string.format('%.4f',self.eta0))
end

