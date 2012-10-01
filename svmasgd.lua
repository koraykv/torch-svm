--[[
	Implementation of some simple linear classifiers to test with 
	Leon Bottou's implementation for comparing in terms of speed
	and convergence properties.

	hehe, this seems like ~10 times slower
]]--

local svmasgd = torch.class('svm.SvmAsgd')

function svmasgd:__init(nf,lam)
	-- weights/biases
	self.w = torch.FloatTensor(nf):zero()
	self.a = torch.FloatTensor(nf):zero()
	self.wdiv = 1
	self.adiv = 1
	self.b = 0
	self.ab = 0
	self.wfrac = 0
	-- lambda
	self.lambda = lam
	-- step size
	self.eta0 = 1
	self.mu0 = 1
	-- counter
	self.nupdate = 0
	self.avstart = 1
	self.averaging = false
	-- number of features (since input is sparse)
	self.nf = nf
	-- flags
	self.regbias = false
	self.svmloss = svm.hingeloss
end

function svmasgd:loss(a,y)
	return self.svmloss(a,y)
end

function svmasgd:renorm()
	if self.wdiv ~= 1 or self.adiv ~= 1 or self.wfrac ~= 0 then
		self.a:mul(1/self.adiv)
		self.a:add(self.wfrac/self.adiv, self.w)
		self.w:mul(1/self.wdiv)
		self.wdiv = 1
		self.adiv = 1
		self.wfrac = 0
	end
end

function svmasgd:wnorm()
	local wd = self.w:double()
	local norm = torch.dot(wd,wd) / self.wdiv / self.wdiv
	if self.regbias then
		norm = norm + self.ab*self.ab
	end
	return norm
end

function svmasgd:anorm()
	self:renorm()
	local ad = self.a:double()
	local norm = torch.dot(ad,ad)
	if self.regbias then
		norm = norm + self.ab*self.ab
	end
	return norm
end

function svmasgd:testOne(y,si,sx)

	-- local variables
	local w    = self.w
	local a    = self.a
	local wdiv = self.wdiv
	local adiv = self.adiv
	local wfrac = self.wfrac
	local ab   = self.ab

	local s = svm.spdot(a,si,sx)

	if wfrac ~= 0 then
		s = s + svm.spdot(w,si,sx)*wfrac
	end
	s = s / adiv + ab

	-- calculate gradient from loss
	local lx,ldx = self:loss(s,y)
	local e = 0
	if s*y <= 0 then
		e = 1
	end
	return s,lx,e
end

function svmasgd:trainOne(y,si,sx,eta,mu)
	-- local variables
	local w    = self.w
	local a    = self.a
	local wdiv = self.wdiv
	local adiv = self.adiv
	local wfrac = self.wfrac
	local b    = self.b
	local ab   = self.ab
	local lambda = self.lambda

	if adiv > 1e5 or wdiv > 1e5 then
		self:renorm()
		adiv = self.adiv
		wdiv = self.wdiv
		wfrac = self.wfrac
	end

	local s = svm.spdot(w,si,sx)/wdiv + b

	-- update wdiv
	wdiv = wdiv / (1 - eta*lambda)

	-- calculate gradient from loss
	local lx,ldx = self:loss(s,y)

	local etd = eta * ldx * wdiv

	-- update weights
	if etd ~= 0 then
		svm.spadd(w,etd,si,sx)
	end

	-- averaging
	if mu >= 1 then
		if self.averaging then
			a:zero()
		end
		adiv = wdiv
		wfrac = 1
	elseif mu > 0 then
		if etd ~= 0 then
			svm.spadd(a,-wfrac*etd,si,sx)
		end
		adiv = adiv / (1-mu)
		wfrac = wfrac + mu * adiv / wdiv
	end

	-- update bias
	if self.regbias then
		b = b * (1- eta * 0.01 * lambda)
	end
	b = b + eta*0.01*ldx
	ab = ab + mu * (b - ab);
	-- put back modified numbers
	self.adiv = adiv
	self.wdiv = wdiv
	self.wfrac = wfrac
	self.ab = ab
	self.b = b

end

function svmasgd:trainepoch(data)
	print('Training on ' .. data:size() .. ' samples')
	-- local variables
	local lambda = self.lambda
	local eta0 = self.eta0
	local mu0 = self.mu0
	local nupdate = self.nupdate

	-- run over every sample
	for i=1,data:size() do
		local ex = data[i]

		-- update learning rate
		local eta = eta0 / math.pow(1 + lambda*eta0*nupdate,0.75)
		local mu = 1
		if nupdate >= self.avstart then
			if not self.averaging then self.averaging = true end
			mu = mu0 / (1 + mu0 * (nupdate - self.avstart))
			-- print(i)
		end

		-- train for a sample
		self:trainOne(ex[1], ex[2][1], ex[2][2], eta, mu)

		nupdate = nupdate + 1
	end
	io.write('wNorm=' .. string.format('%.2f',self:wnorm()))
	io.write(' aNorm=' .. string.format('%.2f',self:anorm()))
	io.write(' wBias=' .. string.format('%.2f',self.b))
	io.write(' aBias=' .. string.format('%.2f\n',self.ab))
	self.nupdate = nupdate
end


function svmasgd:test(data)

	io.write('Testing on ' .. data:size() .. ' samples\n')

	local loss = 0
	local nerr = 0
	for i=1,data:size() do
		local ex = data[i]
		local y = ex[1]
		local si = ex[2][1]
		local sx = ex[2][2]
		local s,l,e = self:testOne(y,si,sx)
		loss = loss + l
		nerr = nerr + e
	end

	loss = loss/data:size()
	nerr = nerr/data:size()

	io.write('Loss=' .. string.format('%.8f',loss))
	io.write(' Cost=' .. string.format('%.8f',loss + 0.5*self.lambda*self:wnorm()))
	io.write(' Misclassification=' .. string.format('%.2f %%\n',100*nerr))
end

function svmasgd:train(trdata,tedata,epochs)

	self.avstart = self.avstart * trdata:size()
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
	self:renorm()
end

function svmasgd:evalEta(nsample,data,eta)
	-- clone the weight and bias
	local w = self.w:clone()
	local b = self.b
	local wdiv = self.wdiv
	for i=1,nsample do
		local ex = data[i]
		local y = ex[1]
		local si = ex[2][1]
		local sx = ex[2][2]

		self:trainOne(y,si,sx,eta,1.0)
	end
	local loss = 0
	for i=1,nsample do
		local ex = data[i]
		local y = ex[1]
		local si = ex[2][1]
		local sx = ex[2][2]

		local s,l,e = self:testOne(y,si,sx)

		loss = loss + l
	end
	local cost = loss/nsample + 0.5 * self.lambda * self:wnorm()
	self.w:copy(w)
	self.b = b
	self.wdiv = wdiv
	return cost
end

function svmasgd:determineEta0(nsample,data)
	local factor = 2
	local loeta = 1
	local locost = self:evalEta(nsample,data,loeta)
	local hieta = loeta * factor
	local hicost = self:evalEta(nsample,data,hieta)
	-- print('lo='..locost .. ' hi=' .. hicost)
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

