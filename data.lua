
-- write a data/label file in libsvmformatted file.
-- fname : libsvm formatted file name
-- data  : {l,d}
-- d[1] is supposed to be index tensor, d[2] is supposed to be value tensor, where each line is a sample
-- l is supposed to be a vector where each entry is the label
function svm.ascwrite(fname, data)
    print('Writing ' .. fname)
    local function vectostr(i, x)
        local str = {}
        local cntr = 1
        x:apply(function(v)
            table.insert(str, string.format('%d:%g', i[cntr], v))
            cntr = cntr + 1
            return v
        end)
        return table.concat(str, ' ')
    end

    local of = torch.DiskFile(fname, 'w')
    for i = 1, #data do
        local ex = data[i]
        of:writeString(string.format('%+g %s\n', ex[1], vectostr(ex[2][1], ex[2][2])))
    end
    of:close()
end

-- read libsvm formatted data file into a label and data tensor
-- returns two outputs, the data and label
function svm.ascread(fname, zero_based)
    if zero_based == nil then zero_based = false end -- If the indices starts with 0 or 1 (default)

    print('Reading ' .. fname)
    local function readline(line)
        local label = tonumber(string.match(line, '^([%+%-]?%s?%d+)'))
        if not label then
            error('could not read label')
        end
        local vals = {}
        local inds = {}
        local indcntr = zero_based and 0 or 1
        for ind, val in string.gmatch(line, '(%d+):([%+%-]?%d?%.?%d+)') do
            ind = tonumber(ind)
            val = tonumber(val)
            if not ind or not val then
                error('reading failed')
            end
            if ind < indcntr then
                error('indices are not in increasing order')
            end
            table.insert(inds, ind)
            table.insert(vals, val)
            indcntr = indcntr + 1
        end
        return label, { torch.IntTensor(inds), torch.FloatTensor(vals) }
    end

    local data = {}
    local maxdim = 0
    local labels = {}
    setmetatable(labels, { __index = function() return 0 end })
    local minsparse = math.huge
    local maxsparse = 0
    for line in io.lines(fname) do
        local lbl, vals = readline(line)
        table.insert(data, { lbl, vals })
        -- stats
        maxdim = math.max(maxdim, vals[1][-1])

        labels[lbl] = labels[lbl] + 1

        minsparse = math.min(minsparse, vals[1]:size(1))
        maxsparse = math.max(maxsparse, vals[1]:size(1))
    end

    for l, c in pairs(labels) do
        io.write(string.format("# of samples of label %d = %d\n", l, c))
    end

    io.write(string.format("# of total    samples = %d\n", #data))
    io.write(string.format("# of max dimensions   = %d\n", maxdim))
    io.write(string.format("Min # of dims = %d\n", minsparse))
    io.write(string.format("Max # of dims = %d\n", maxsparse))
    return data, maxdim
end


--[[
	A simple dataset table
	If the filename extension is .bin, then 
	it will be assumed to be binary, otherwise it will be assumed
	ascii formatted file.
	The format of the file is svmlight format, for binary format,
	format suggested by Leon Bottou is used.
]]--

function svm.dataset(fname)
    if not paths.filep(fname) then
        error('File does not exist ' .. fname)
    end

    local data, maxdim
    if fname:match('%.bin') then
        data, maxdim = svm.binread(fname, true)
    else
        data, maxdim = svm.ascread(fname)
    end
    local nsamples = #data
    local dataset = {}
    function dataset:size() return nsamples end

    function dataset:nfeature() return maxdim end

    function dataset:data() return data end


    -- be careful , this is just for experimentation, it will be very very very slooooooow.
    local dense = false
    function dataset:dense()
        dense = true
    end

    local dx
    local function todense(ind, x)
        dx = dx or torch.FloatTensor(maxdim)
        dx:zero()
        for i = 1, ind:size(1) do
            dx[ind[i]] = x[i]
        end
        return { nil, dx }
    end

    setmetatable(dataset, {
        __index = function(self, i)
            local ind = math.mod(i - 1, nsamples) + 1
            if dense then
                local ex = data[ind]
                return { ex[1], todense(ex[2][1], ex[2][2]) }
            else
                return data[ind]
            end
        end
    })

    return dataset
end
