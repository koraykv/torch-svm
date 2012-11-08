

function svm.hingeloss(a,y)
	local z = a*y
	if z > 1 then return 0,0 end
	return 1-z,y
end

function svm.logloss(a,y)
	local z = a*y
	if z>18 then
		return math.exp(-z), y*math.exp(-z)
	end
	if z<-18 then
		return -z,y
	end
	return math.log(1+math.exp(-z)),y/(1+math.exp(z))
end

function svm.squaredhingeloss(a,y)
	local z = a*y
	if z > 1 then return 0,0 end
	local d=1-z
	return 0.5*d*d,y*d
end


