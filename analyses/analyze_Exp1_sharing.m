% author: Andrew Chang, ac8888@nyu.edu, Nov 4, 2022

%% Due to privacy issue, the participants with cognitive, developmental, neurological, psychiatric, or speech-language disorders were excluded here, and their text responses were removed.
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')


%dataQues = readtable('exp1/data/judgeSpeechMusic_20210407_logNormShiftAmp_May+11,+2021_09.39.xlsx');
%dataQues = clinSubjectRemoval(dataQues);

%save('exp1/data/dataQues_exp1','dataQues')


%% load questionnaire and calculate the Gold-MSI scores

load('exp1/data/dataQues_exp1')

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


listing = dir('exp1/data/*.csv');


listing2 = listing;

% exclude the participats who did not wear headphones or complete the task
badPart = zeros(1,length(listing));
for nFile = 1:length(listing)
    tempSubNum = str2double(listing(nFile).name(1:end-70));
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

%% calculate the response


dataAll = addvars(dataAll,NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),10),...
    'NewVariableNames',{'probeACC','totalBias','resConsis_subj','resConsis_cdf','percResp'});

for nFile = 1:length(listing2)

    data = readtable(['exp1/data/',listing2(nFile).name]);
    
    if height(data) < 300 % skip the files which did not complete the task
        continue
    elseif max(data.expTrials_thisN) ~= 300-1 % skip the files which did not complete the final trial
        continue
    end
    
    subNum(nFile) = str2double(listing2(nFile).name(1:end-70));
    
    dataAll_ind = find(contains(dataAll.id,num2str(subNum(nFile))));

    %% add peak Hz to the table
    peakHz = NaN(size(data,1),1);
    data = addvars(data,peakHz);

    peakHzStr = ["AM0.6Hz"; "AM1.2Hz"; "AM1.8Hz"; "AM2.4Hz"; "AM3Hz"; "AM3.6Hz"; "AM4.2Hz"; "AM4.8Hz"; "AM5.4Hz"; "AM6Hz"];

    for n = 1:length(peakHzStr)
        TF = contains(data.audioFile,peakHzStr(n));
        if sum(TF)~=30 % there should be 30 trials per peak Hz condition
            disp('something is wrong!!!')
        end
        data.peakHz(TF) = n*0.6;
    end

    %% Check probe ACC

    probeACC = mean(data.beepResp_corr(~isnan(data.beepResp_keys)));

    %% Check test-retest reliability

    C = unique(data.audioFile);
    C(strlength(C)==0) = []; % remove the empty row

    totalBias = sum(data.ratingSlider_response==2)/(length(C)*2); % the overall response bias

    resConsis = NaN(1,length(C));

    for m = 1:length(C)
        resPair = data.ratingSlider_response(string(data.audioFile)==C(m));
        if length(resPair) > 2
            disp('something is wrong!!!')
        end
        if resPair(1)==resPair(2)
            resConsis(m) = 1;
        else
            resConsis(m) = 0;
        end
    end

    resConsis_subj = mean(resConsis); % mean response consistency across 150 trials

    pBino = totalBias^2+(1-totalBias)^2; % p(consistency) = p(bias)^2 + (1-p(bias))^2
    resConsis_cdf = cdf('Binomial',sum(resConsis),length(C),pBino); % the CDF of response consistency, under binomial distribution
    
    
    %% get the percentage responses

    for n = 1:10
        percResp(n) = mean(data.ratingSlider_response(data.peakHz==n*0.6));
    end
    
    rs = 10;
    for n = 1:10
        tempData = data.ratingSlider_response(data.peakHz==n*0.6);
    end
    
    dataAll.probeACC(dataAll_ind)       = probeACC;
    dataAll.totalBias(dataAll_ind)      = totalBias;
    dataAll.resConsis_subj(dataAll_ind) = resConsis_subj;
    dataAll.resConsis_cdf(dataAll_ind)  = resConsis_cdf;
    dataAll.percResp(dataAll_ind,:)     = percResp;

    
    
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




%% regression

r2 = [];
p_fit = [];
slope = [];
fittedLine = [];

for n = 1:height(dataAll)
    x = dataAll.percResp(n,:);
    mld = fitlm(0.6:0.6:6,x);
    r2(n) = mld.Rsquared.Ordinary;
    p_fit(n) = coefTest(mld);
    slope(n) = mld.Coefficients.Estimate(2); % slope coef
    fittedLine(n,:) = mld.Fitted;
end




[~,p,~,stats] = ttest(slope)
abs(mean(slope)/std(slope))



%% correlation between slope and musical sophistication score

[r_MSIgen,p_MSIgen] = corr(slope',dataAll.gen)
[r_MSIgen,p_MSIgen] = corr(slope(dataAll.gen~=34)',dataAll.gen(dataAll.gen~=34)) % excluding 1 outlier

% split the data at slope = 0
[~,p,~,stat] = ttest2(dataAll.gen(slope>0),dataAll.gen(slope<0),'vartype','unequal')


sp = sqrt( ((sum(slope<0)-1)*std(dataAll.gen(slope<0))^2 + (sum(slope>0)-1)*std(dataAll.gen(slope>0))^2) / (length(dataAll.gen)-2) );

cohenD = abs((mean(dataAll.gen(slope<0))-mean(dataAll.gen(slope>0)))/sp)


%% plot

figure('Position', [10 10 1500 500])
subplot(1,10,[1,1.25])
imagesc(0.6:0.6:6, 1:length(dataPlot) ,dataPlot)
yticks([])
xticks([0.6:0.6:6])
xticklabels({'0.6','','','','','','','','','6.0'})
ylabel('participants')
xlabel('peak AM frequency (Hz)')
% set(gca, 'XTick',xtnew, 'XTickLabel',xtlbl) 
cb = colorbar('Ticks',[1,2], 'TickLabels',{'music','speech'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;


subplot(1,10,[2.5,4.5])
dataPlot = dataAll.percResp;

p = plot(0.6:0.6:6,fittedLine,'color',[col(4,:),0.35], 'LineWidth', 1);
hold on
p = plot(0.6:0.6:6,mean(fittedLine),'color','k', 'LineWidth', 2);
xlabel('peak AM frequency (Hz)')
ylabel('response')
xticks(0.6:0.6:6)
xticklabels({'','1.2','','2.4','','3.6','','4.8','','6.0'})
xlim([0.6,6])
yticks([1,2])
yticklabels({'music','speech'})
ylim([1,2])
ytickangle(90)
% xtickangle(45)
set(gca,'fontsize',14)

% hold on
% col = lines(4);
% h1 = shadedErrorBar(0.6:0.6:6,mean(dataPlot),std(dataPlot)/sqrt(size(dataPlot,1)),{'*-','color',col(4,:), 'LineWidth', 2},0.5);
% xlabel('peak AM frequency (Hz)')
% ylabel('response')
% xticks(0.6:0.6:6)
% xticklabels({'0.6','','','2.4','','','4.2','','','6.0'})
% xlim([0.6,6])
% yticks([1,1.25,1.5,1.75,2])
% yticklabels({'music','','','','speech'})
% ylim([1,2])
% set(gca,'fontsize',14)
% grid on
% box on





% subplot(1,3,2)
subplot(1,10,[6,6.5])
bar(1,mean(slope),'facecolor',col(4,:),'LineWidth',2);
hold on
er = errorbar(1,mean(slope),std(slope)/sqrt(length(slope)),'LineWidth',2,'CapSize',20);   
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
scatter(ones(length(slope),1)*1,slope,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
xticks([])
ylabel('response slope (regression coefficient)')
set(gca,'fontsize',14)
xlim([0.4,1.6])



subplot(1,10,[8,10])
% subplot(1,3,3);
scatter(slope(dataAll.gen~=34),dataAll.gen(dataAll.gen~=34),'MarkerEdgeColor',col(4,:),'LineWidth',2);xlabel('response slope');ylabel('General Musical Sophistication');set(gca,'fontsize',14);ylim([18,126]);h=lsline;h.Color='k';h.LineWidth=1;box on;...
hold on; scatter(slope(dataAll.gen==34),dataAll.gen(dataAll.gen==34),'MarkerEdgeColor',[0.7,0.7,0.7],'LineWidth',2)


%% new plot
figure
p = plot(0.6:0.6:6,fittedLine,'color',[col(4,:),0.35], 'LineWidth', 1);
hold on
p = plot(0.6:0.6:6,mean(fittedLine),'color','k', 'LineWidth', 2);
% for i=1:numel(p)
%     c = get(p(i), 'Color');
%     set(p(i), 'Color', [c 0.5]);
% end
hold on
% scatter(0.6:0.6:6,dataPlot,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7)
xlabel('peak AM frequency (Hz)')
ylabel('response')
xticks(0.6:0.6:6)
xticklabels({'0.6','','','2.4','','','4.2','','','6.0'})
xlim([0.6,6])
yticks([1,1.25,1.5,1.75,2])
yticklabels({'music','','','','speech'})
ylim([1,2])
set(gca,'fontsize',14)

%% power estimation

statsPower = 0.8;
sampsizepwr('t',[mean(slope),std(slope)],0,0.8,[],'Alpha',0.05)

