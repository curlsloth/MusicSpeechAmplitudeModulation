function data = clinSubjectRemoval(data)
%replace the clinical conditions by Yes/No
    data.clinical = zeros(size(data,1),1);
    clinSubj = [];
    for n = 1:height(data)
        tempstr  = [data.Q6{n},'//',data.Q7{n}];
        if contains(tempstr,{'slex';'AD';'epression';'nixety';'ipolar';'isorder';'psych';'speech';'ysgraphia';'lisp';'ear';'tutter';'ankyloglossia';'chiari malformation';'adhd';'add';'chronic migraines';'diagnos'})
            if ~contains(tempstr,'no relevant disorder')
                disp(['***',tempstr,'***'])
                data.clinical(n) = 1;
                %clinSubj(end+1) = n;
            end
        end
        
    end
    data.Q6=[];
    data.Q7=[];
    data(data.clinical==1,:) = [];
end