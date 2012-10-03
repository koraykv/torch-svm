require 'libsvm'

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

-- write a data/label file in libsvmformatted file.
-- fname : libsvm formatted file name
-- data  : {l,d}
-- d[1] is supposed to be index tensor, d[2] is supposed to be value tensor, where each line is a sample
-- l is supposed to be a vector where each entry is the label
function svm.ascwrite(fname,data)
	print('Writing ' .. fname)
	local function vectostr(i,x)
		local str = {}
		local cntr = 1
		x:apply(function(v) 
			table.insert(str,string.format('%d:%g', i[cntr], v))
			cntr = cntr + 1
			return v
			end)
		return table.concat(str, ' ')
	end

	local of = torch.DiskFile(fname,'w')
	for i=1,#data do
		local ex = data[i]
		of:writeString(string.format('%+d %s\n', ex[1], vectostr(ex[2][1],ex[2][2])))
	end
	of:close()
end

-- read libsvm formatted data file into a label and data tensor
-- returns two outputs, the data and label
function svm.ascread(fname)
	print('Reading ' .. fname)
	local function readline(line)
		local label = tonumber(string.match(line,'^([%+%-]?%s?%d+)'))
		if not label then
			error('could not read label')
		end
		if label ~= 1 and label ~=-1 then
			error('label has to be +1 or -1')
		end
		local vals = {}
		local inds = {}
		local indcntr = 0
		for ind,val in string.gmatch(line,'(%d+):([%+%-]?%d?%.?%d+)') do
			indcntr = indcntr + 1
			ind = tonumber(ind)
			val = tonumber(val)
			if not ind or not val then
				error('reading failed')
			end
			if ind < indcntr then
				error('indices are not in increasing order')
			end
			table.insert(inds,ind)
			table.insert(vals,val)
		end
		return label,{torch.IntTensor(inds),torch.FloatTensor(vals)}
	end
	local data = {}
	local maxdim = 0
	local npos = 0
	local nneg = 0
	local minsparse = math.huge
	local maxsparse = 0
	for line in io.lines(fname) do
		local lbl,vals = readline(line)
		table.insert(data,{lbl,vals})
		-- stats
		maxdim = math.max(maxdim,vals[1][-1])
		if lbl == 1 then npos = npos + 1 else nneg = nneg + 1 end
		minsparse = math.min(minsparse,vals[1]:size(1))
		maxsparse = math.max(maxsparse,vals[1]:size(1))
	end
	io.write(string.format("# of positive samples = %d\n",npos))
	io.write(string.format("# of negative samples = %d\n",nneg))
	io.write(string.format("# of total    samples = %d\n",#data))
	io.write(string.format("# of max dimensions   = %d\n",maxdim))
	io.write(string.format("Min # of dims = %d\n",minsparse))
	io.write(string.format("Max # of dims = %d\n",maxsparse))
	return data,maxdim
end

include('svmsgd.lua')
include('svmasgd.lua')
include('data.lua')

