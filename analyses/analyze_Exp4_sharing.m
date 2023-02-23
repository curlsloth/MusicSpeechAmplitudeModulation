% author: Andrew Chang, ac8888@nyu.edu, Nov 4, 2022

%% Due to privacy issue, the participants with cognitive, developmental, neurological, psychiatric, or speech-language disorders were excluded here, and their text responses were removed.
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')


%dataQues = readtable('exp4/data/judgeSpeechMusic_20210919_logNormShiftAmp_detection_fixFreq_October+12,+2021_10.52.xlsx');
%dataQues = clinSubjectRemoval(dataQues);

%save('exp4/data/dataQues_exp4','dataQues')


%% load questionnaire and calculate the Gold-MSI scores

load('exp4/data/dataQues_exp4')

codeDir = [1,1,1,1,1,1,1,1,0,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1]; % 1=positive/0=negative
ind_GoldMSI = find(contains(string(dataQues.Properties.VariableNames),{'Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28'}));

if length(codeDir) ~= length(ind_GoldMSI)
    disp('***** something is wrong with ind_GoldMSI *****')
end

dataGoldMSI = table2array(dataQues(:,ind_GoldMSI));
dataGoldMSI2 = dataGoldMSI;
dataGoldMSI2(:,codeDir==0) = 8-dataGoldMSI(:,codeDir==0);

% select the subscales
actEng_ind = [1,3,8,15,21,24,28,34,38];
perAbi_ind = [5,6,11,12,13,18,22,23,26];
musTra_ind = [14,27,32,33,35,36,37];
sinAbi_ind = [4,7,10,17,25,29,30];
emo_ind = [2,9,16,19,20,31];
gen_ind = [1,3,15,24,12,23,14,27,32,33,37,4,7,10,17,25,29,19];

actEng = sum(dataGoldMSI2(:,actEng_ind),2);
perAbi = sum(dataGoldMSI2(:,perAbi_ind),2);
musTra = sum(dataGoldMSI2(:,musTra_ind),2);
sinAbi = sum(dataGoldMSI2(:,sinAbi_ind),2);
emo = sum(dataGoldMSI2(:,emo_ind),2);
gen = sum(dataGoldMSI2(:,gen_ind),2);

% add subscores to the data table
dataQues = addvars(dataQues,actEng,perAbi,musTra,sinAbi,emo,gen);


%% read file names


listing = dir('exp4/data/*.csv');


listing2 = listing;

% exclude the participats who did not wear headphones or complete the task
badPart = zeros(1,length(listing));
for nFile = 1:length(listing)
    tempSubNum = str2double(listing(nFile).name(1:5));
    tempQuesInd = ismember(str2double(dataQues.id),tempSubNum);
    if sum(tempQuesInd) == 0 % if someone did not complete both questionnaire and the task
        badPart(nFile) = 1; % don't analyze the data
    end
    if dataQues.Q10(tempQuesInd)==2 % if someone report not wearing headphones
        disp(num2str(nFile))
        badPart(nFile) = 1;
    end
        
end


listing2(boolean(badPart)) = [];

dataAll = dataQues;

%%

warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')

dataAll = addvars(dataAll,NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),5),NaN(height(dataAll),5),...
    'NewVariableNames',{'probeACC','totalBias','totalBiasM','totalBiasS','resConsis_subj','resConsis_cdf','percResp_Music','percResp_Speech'});

for nFile = 1:length(listing2)

    data = readtable(['exp4/data/',listing2(nFile).name]);
    
    if height(data) < 300 % skip the files which did not complete the task
        continue
    end
    
    subNum(nFile) = str2double(listing2(nFile).name(1:5));
    
    dataAll_ind = find(contains(dataAll.id,num2str(subNum(nFile))));

    %% add peak Hz and S to the table
    peakHz = NaN(size(data,1),1);
    peakS = NaN(size(data,1),1);
    blockType = string(NaN(size(data,1),1));
    data = addvars(data,peakHz,peakS,blockType);

    peakHzStr = ["AM2Hz"];
    peakSStr = ["s15";"s25";"s35";"s45";"s55"];
    peakHzNum = [2];
    peakSNum = [15,25,35,45,55]/100;


    for n = 1:length(peakHzStr)
        TF = contains(data.audioFile,peakHzStr(n));
        data.peakHz(TF) = peakHzNum(n);
    end
    
	for n = 1:length(peakSStr)
        TF = contains(data.audioFile,peakSStr(n));
        data.peakS(TF) = peakSNum(n);
    end
    
    
    blockName_ind = find(~cellfun(@isempty,data.blockName)); % index the non-empty cell array
    data.blockType(1:blockName_ind(1)) = data.blockName{blockName_ind(1)}(1);
    data.blockType(blockName_ind(1)+1:blockName_ind(2)) = data.blockName{blockName_ind(2)}(1);
    data.blockType(blockName_ind(2)+1:blockName_ind(3)) = data.blockName{blockName_ind(3)}(1);
    data.blockType(blockName_ind(3)+1:blockName_ind(4)) = data.blockName{blockName_ind(4)}(1);

    

    %% Check probe ACC

    probeACC = mean(data.beepResp_corr(~isnan(data.beepResp_keys)));

    %% Check test-retest reliability

    C = unique(data.audioFile);
    C(strlength(C)==0) = []; % remove the empty row

    totalBias = sum(data.ratingSlider_response==2)/sum(~isnan(data.ratingSlider_response)); % the overall response bias


    totalBiasM = sum(data.ratingSlider_response==2 & strcmp(data.blockType,'M'))/sum(~isnan(data.ratingSlider_response) & strcmp(data.blockType,'M')); % the overall response bias
    totalBiasS = sum(data.ratingSlider_response==2 & strcmp(data.blockType,'S'))/sum(~isnan(data.ratingSlider_response) & strcmp(data.blockType,'S')); % the overall response bias


    resConsis = [];
    
    for b = ['M','S']
        for m = 1:length(C)
            resPair = data.ratingSlider_response(string(data.audioFile)==C(m) & strcmp(data.blockType,b));
            if length(resPair) > 2
                disp('something is wrong!!!')
            end
            if resPair(1)==resPair(2)
                resConsis(end+1) = 1;
            else
                resConsis(end+1) = 0;
            end
        end
    end

    resConsis_subj = mean(resConsis); % mean response consistency across 150 trials

    pBino = totalBias^2+(1-totalBias)^2; % p(consistency) = p(bias)^2 + (1-p(bias))^2
    resConsis_cdf = cdf('Binomial',sum(resConsis),length(resConsis),pBino); % the CDF of response consistency, under binomial distribution
    
    
    %% get the percentage responses

    for n = 1:length(peakSNum)
        percResp_Music(n) = mean(data.ratingSlider_response(data.peakS==peakSNum(n)  &   strcmp(data.blockType,'M')));
        percResp_Speech(n) = mean(data.ratingSlider_response(data.peakS==peakSNum(n)  &   strcmp(data.blockType,'S')));
    end
    

    
    dataAll.probeACC(dataAll_ind)       = probeACC;
    dataAll.totalBias(dataAll_ind)      = totalBias;
    dataAll.totalBiasM(dataAll_ind)      = totalBiasM;
    dataAll.totalBiasS(dataAll_ind)      = totalBiasM;
    dataAll.resConsis_subj(dataAll_ind) = resConsis_subj;
    dataAll.resConsis_cdf(dataAll_ind)  = resConsis_cdf;
    dataAll.percResp_Music(dataAll_ind,:)     = percResp_Music;
    dataAll.percResp_Speech(dataAll_ind,:)    = percResp_Speech;
    
    
end

figure;
histogram(dataAll.resConsis_cdf,0:0.05:1,'normalization','probability')
xlabel('CDF of response consistency')
ylabel('% of participants')
xticks(0:0.1:1)
set(gca,'fontsize',16)


%% exclude participants


dataAll(dataAll.Progress~=100,:) = []; % who did not finish
dataAll(isnan(dataAll.probeACC),:) = []; % who did not finish
dataAll(dataAll.totalBias==1,:) = []; % whose responses were all 1
dataAll(dataAll.totalBias==0,:) = []; % whose responses were all 0
dataAll(dataAll.probeACC<0.9,:) = []; % whose accuracy on the probe trials lower than 90%

% response bias exceeding 50 +/- 15%
dataAll(abs(dataAll.totalBiasM-0.5)>0.15,:) = [];
dataAll(abs(dataAll.totalBiasS-0.5)>0.15,:) = [];



%% regression


dataM = dataAll.percResp_Music;
dataS = dataAll.percResp_Speech;


for n = 1:height(dataAll)
    xM = dataM(n,:)';
    mld = fitlm(0.15:0.1:0.55,xM);
    r2_M(n) = mld.Rsquared.Ordinary;
    pM_fit(n) = coefTest(mld);
    slope_M(n) = mld.Coefficients.Estimate(2);
    fittedLine_M(n,:) = mld.Fitted;
    
    
    xS = dataS(n,:)';
    mld = fitlm(0.15:0.1:0.55,xS);
    r2_S(n) = mld.Rsquared.Ordinary;
    pS_fit(n) = coefTest(mld);
    slope_S(n) = mld.Coefficients.Estimate(2);
    fittedLine_S(n,:) = mld.Fitted;

end


[~,p,~,stat] = ttest(slope_M)
[~,p,~,stat] = ttest(slope_S)

abs(mean(slope_M)/std(slope_M))
abs(mean(slope_S)/std(slope_S))

mean([r2_M,r2_S])
std([r2_M,r2_S])/sqrt(length([r2_M,r2_S]))


%% correlation between slope and musical sophistication score

[rM_MSIgen,pM_MSIgen] = corr(slope_M',dataAll.gen)
[rS_MSIgen,pS_MSIgen] = corr(slope_S',dataAll.gen)



%% new plot



col = lines(7);


figure('Position', [10 10 1200 600])


% Music
subplot(2,10,[1,1.15])
imagesc(peakSNum, 1:length(dataM) ,dataM, [1 2])
yticks([])
xticks(peakSNum)
xticklabels({'0.15','','','','0.55'})
ylabel('participants')
xlabel('\sigma')
cb = colorbar('Ticks',[1,2], 'TickLabels',{'others','music'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;
cb.Label.String = 'response';

subplot(2,10,[2.5,4])
p = plot(peakSNum,fittedLine_M,'color',[col(6,:),0.35], 'LineWidth', 1);
hold on
p = plot(peakSNum,mean(fittedLine_M),'color','k', 'LineWidth', 2);
xlabel('\sigma')
ylabel('response')
xticks(peakSNum)
xlim([0.15,0.55])
xtickangle(45)
yticks([1,2])
yticklabels({'others','music'})
ylim([1,2])
ylabel('response')
ytickangle(90)
set(gca,'fontsize',14)
title('Music')
% grid on
box on

subplot(2,10,[9,10])
scatter(slope_M,dataAll.gen,100,'filled','MarkerFaceColor',col(6,:),'MarkerFaceAlpha',.7);xlabel('response slope');ylabel('General Musical Sophistication');title('Music');set(gca,'fontsize',14);ylim([18,126]);box on;



% Speech
subplot(2,10,[1,1.15]+10)
imagesc(peakSNum, 1:length(dataS) ,dataS, [1 2])
yticks([])
xticks(peakSNum)
xticklabels({'0.15','','','','0.55'})
ylabel('participants')
xlabel('\sigma')
cb = colorbar('Ticks',[1,2], 'TickLabels',{'others','speech'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;
cb.Label.String = 'response';

subplot(2,10,[2.5,4]+10)
p = plot(peakSNum,fittedLine_S,'color',[col(7,:),0.35], 'LineWidth', 1);
hold on
p = plot(peakSNum,mean(fittedLine_S),'color','k', 'LineWidth', 2);
xlabel('\sigma')
ylabel('response')
xticks(peakSNum)
xlim([0.15,0.55])
xtickangle(45)
yticks([1,2])
yticklabels({'others','speech'})
ylim([1,2])
ylabel('response')
ytickangle(90)
set(gca,'fontsize',14)
title('Speech')
% grid on
box on

subplot(2,10,[9,10]+10)
scatter(slope_S,dataAll.gen,100,'filled','MarkerFaceColor',col(7,:),'MarkerFaceAlpha',.7);xlabel('response slope');ylabel('General Musical Sophistication');title('Speech');set(gca,'fontsize',14);ylim([18,126]);box on;




subplot(1,10,[5.5,7.5])
bar(1,mean(slope_M),'facecolor',col(6,:),'LineWidth',2);hold on
bar(2,mean(slope_S),'facecolor',col(7,:),'LineWidth',2);
er = errorbar([1,2],[mean(slope_M),mean(slope_S)],...
    [std(slope_M),std(slope_S)]/sqrt(length(slope_M)),'LineWidth',2,'CapSize',20);   
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
scatter(ones(length(slope_M),1),slope_M,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
scatter(ones(length(slope_S),1)*2,slope_S,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
xticks([1,2])
xlim([0.25,2.75])
xticklabels({'Music','Speech'})
ylabel('response slope (regression coefficient)')
set(gca,'fontsize',14)


%% estimate required N

sampsizepwr('t',[mean(slope_M),std(slope_M)],0,0.8,[],'Alpha',0.05)
sampsizepwr('t',[mean(slope_S),std(slope_S)],0,0.8,[],'Alpha',0.05)


%% fit logistic function


a_M = [];
b_M = [];
r2_logistic_M = [];
for n = 1:size(dataM,1)
    [xData, yData] = prepareCurveData( peakSNum, dataM(n,:)-1 );
    
    % Set up fittype and options.
    ft = fittype( '1/(1+exp(-b*(x-a)))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Robust = 'LAR';
    opts.StartPoint = [median(peakSNum) 0.5];
    opts.Lower = [min(peakSNum) -Inf];
    opts.Upper = [max(peakSNum) Inf];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a_M(n) = fitresult.a;
    b_M(n) = fitresult.b;
    r2_logistic_M(n) = gof.rsquare;


end


a_S = [];
b_S = [];
r2_logistic_S = [];
for n = 1:size(dataS,1)
    [xData, yData] = prepareCurveData( peakSNum, dataS(n,:)-1 );
    
    % Set up fittype and options.
    ft = fittype( '1/(1+exp(-b*(x-a)))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Robust = 'LAR';
    opts.StartPoint = [median(peakSNum) 0.5];
    opts.Lower = [min(peakSNum) -Inf];
    opts.Upper = [max(peakSNum) Inf];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a_S(n) = fitresult.a;
    b_S(n) = fitresult.b;
    r2_logistic_S(n) = gof.rsquare;


end

[~,p, ~, stat] = ttest(b_M)
[~,p, ~, stat] = ttest(b_S)
nanmean([r2_logistic_M,r2_logistic_S]<0)
nanmean([r2_logistic_M,r2_logistic_S]-[r2_M,r2_S])
nanmedian([r2_logistic_M,r2_logistic_S]-[r2_M,r2_S])

nanmedian([r2_logistic_M, r2_logistic_S]-[r2_M,r2_S])

[~,p, ~, stat] = ttest([r2_logistic_M, r2_logistic_S], [r2_M,r2_S])
