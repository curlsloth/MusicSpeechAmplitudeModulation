% author: Andrew Chang, ac8888@nyu.edu, Nov 4, 2022

%% Due to privacy issue, the participants with cognitive, developmental, neurological, psychiatric, or speech-language disorders were excluded here, and their text responses were removed.
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')


%dataQues = readtable('exp2/data/judgeSpeechMusic_20210519_logNormShiftAmp_mXs_June+29,+2021_18.34.xlsx');
%dataQues = clinSubjectRemoval(dataQues);

%save('exp2/data/dataQues_exp2','dataQues')

%% load questionnaire and calculate the Gold-MSI scores

load('exp2/data/dataQues_exp2')

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


listing = dir('exp2/data/*.csv');


listing2 = listing;

% exclude the participats who did not wear headphones or complete the task
badPart = zeros(1,length(listing));
for nFile = 1:length(listing)
    tempSubNum = str2double(listing(nFile).name(1:end-74));
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


dataAll = addvars(dataAll,NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),1),NaN(height(dataAll),5),NaN(height(dataAll),5),NaN(height(dataAll),5),...
    'NewVariableNames',{'probeACC','totalBias','resConsis_subj','resConsis_cdf','percResp_peakHz1','percResp_peakHz2','percResp_peakHz3'});

for nFile = 1:length(listing2)

    data = readtable(['exp2/data/',listing2(nFile).name]);
    
    if height(data) < 300 % skip the files which did not complete the task
        continue
    elseif max(data.expTrials_thisN) ~= 300-1 % skip the files which did not complete the final trial
        continue
    end
    
    subNum(nFile) = str2double(listing2(nFile).name(1:end-74));
    
    dataAll_ind = find(contains(dataAll.id,num2str(subNum(nFile))));

    %% add peak Hz and S to the table
    peakHz = NaN(size(data,1),1);
    peakS = NaN(size(data,1),1);
    data = addvars(data,peakHz,peakS);

    peakHzStr = ["AM1Hz"; "AM2.5Hz"; "AM4Hz"];
    peakSStr = ["s15"; "s25"; "s35"; "s45"; "s55"];
    peakHzNum = [1, 2.5, 4];


    for n = 1:length(peakHzStr)
        TF = contains(data.audioFile,peakHzStr(n));
        if sum(TF)~=100 % there should be 30 trials per peak Hz condition
            disp('something is wrong!!!')
        end
        data.peakHz(TF) = peakHzNum(n);
    end
    
	for n = 1:length(peakSStr)
        TF = contains(data.audioFile,peakSStr(n));
        if sum(TF)~=60 % there should be 30 trials per peak Hz condition
            disp('something is wrong!!!')
        end
        data.peakS(TF) = 5+n*10;
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

    for n = 1:5
        percResp_peakHz1(n) = mean(data.ratingSlider_response(data.peakHz==1    &  data.peakS==5+n*10));
        percResp_peakHz2(n) = mean(data.ratingSlider_response(data.peakHz==2.5  &  data.peakS==5+n*10));
        percResp_peakHz3(n) = mean(data.ratingSlider_response(data.peakHz==4    &  data.peakS==5+n*10));
    end
    
   
    dataAll.probeACC(dataAll_ind)       = probeACC;
    dataAll.totalBias(dataAll_ind)      = totalBias;
    dataAll.resConsis_subj(dataAll_ind) = resConsis_subj;
    dataAll.resConsis_cdf(dataAll_ind)  = resConsis_cdf;
    dataAll.percResp_peakHz1(dataAll_ind,:)     = percResp_peakHz1;
    dataAll.percResp_peakHz2(dataAll_ind,:)     = percResp_peakHz2;
    dataAll.percResp_peakHz3(dataAll_ind,:)     = percResp_peakHz3;
    
    
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


data1 = dataAll.percResp_peakHz1;
data2 = dataAll.percResp_peakHz2;
data3 = dataAll.percResp_peakHz3;


for n = 1:height(dataAll)
    
    x1 = data1(n,:);
    x2 = data2(n,:);
    x3 = data3(n,:);
    
    mld = fitlm(0.15:0.1:0.55,x1);
    r2_1(n) = mld.Rsquared.Ordinary;
    p1_fit(n) = coefTest(mld);
    slope_1(n) = mld.Coefficients.Estimate(2);
    fittedLine_1(n,:) = mld.Fitted;
    
    mld = fitlm(0.15:0.1:0.55,x2);
    r2_2(n) = mld.Rsquared.Ordinary;
    p2_fit(n) = coefTest(mld);
    slope_2(n) = mld.Coefficients.Estimate(2);
    fittedLine_2(n,:) = mld.Fitted;
    
    mld = fitlm(0.15:0.1:0.55,x3);
    r2_3(n) = mld.Rsquared.Ordinary;
    p3_fit(n) = coefTest(mld);
    slope_3(n) = mld.Coefficients.Estimate(2);
    fittedLine_3(n,:) = mld.Fitted;
    
end


[~,p,~,stat] = ttest(slope_1)
[~,p,~,stat] = ttest(slope_2)
[~,p,~,stat] = ttest(slope_3)


abs(mean(slope_1)/std(slope_1))
abs(mean(slope_2)/std(slope_2))
abs(mean(slope_3)/std(slope_3))

mean([r2_1,r2_2,r2_3])
std([r2_1,r2_2,r2_3])/sqrt(length([r2_1,r2_2,r2_3]))

%% correlation between slope and musical sophistication score
[r_MSIgen,p_MSIgen] = corr(slope_1',dataAll.gen)
[r_MSIgen,p_MSIgen] = corr(slope_2',dataAll.gen)
[r_MSIgen,p_MSIgen] = corr(slope_3',dataAll.gen)



%% new plot



col = lines(4);


figure('Position', [10 10 1200 800])


% 1 Hz
subplot(3,10,[1,1.15])
imagesc(0.15:0.1:0.55, 1:length(data1) ,data1, [1 2])
yticks([])
xticks([0.15:0.1:0.55])
xticklabels({'0.15','','','','0.55'})
ylabel('participants')
xlabel('\sigma')
cb = colorbar('Ticks',[1,2], 'TickLabels',{'music','speech'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;
cb.Label.String = 'response';

subplot(3,10,[2.5,4])
p = plot(0.15:0.1:0.55,fittedLine_1,'color',[col(1,:),0.35], 'LineWidth', 1);
hold on
p = plot(0.15:0.1:0.55,mean(fittedLine_1),'color','k', 'LineWidth', 2);
xticks(0.15:0.1:0.55)
xlim([0.15,0.55])
xtickangle(45)
yticks([1,2])
yticklabels({'music','speech'})
% xlabel('\sigma')
ylim([1,2])
ylabel('response')
ytickangle(90)
set(gca,'fontsize',12)
title('1 Hz')
% grid on
box on

subplot(3,10,[9,10])
scatter(slope_1,dataAll.gen,100,'filled','MarkerFaceColor',col(1,:),'MarkerFaceAlpha',.7);title('1 Hz');set(gca,'fontsize',14);ylim([18,126]);box on;



% 2.5 Hz
subplot(3,10,[1,1.15]+10)
imagesc(0.15:0.1:0.55, 1:length(data2) ,data2, [1 2])
yticks([])
xticks([0.15:0.1:0.55])
xticklabels({'0.15','','','','0.55'})
ylabel('participants')
xlabel('\sigma')
cb = colorbar('Ticks',[1,2], 'TickLabels',{'music','speech'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;
cb.Label.String = 'response';

subplot(3,10,[2.5,4]+10)
p = plot(0.15:0.1:0.55,fittedLine_2,'color',[col(2,:),0.35], 'LineWidth', 1);
hold on
p = plot(0.15:0.1:0.55,mean(fittedLine_2),'color','k', 'LineWidth', 2);
xticks(0.15:0.1:0.55)
xlim([0.15,0.55])
xtickangle(45)
% xlabel('\sigma')
yticks([1,2])
yticklabels({'music','speech'})
ylim([1,2])
ylabel('response')
ytickangle(90)
set(gca,'fontsize',12)
title('2.5 Hz')
% grid on
box on

subplot(3,10,[9,10]+10)
scatter(slope_2,dataAll.gen,100,'filled','MarkerFaceColor',col(2,:),'MarkerFaceAlpha',.7);ylabel('General Musical Sophistication');title('2.5 Hz');set(gca,'fontsize',14);ylim([18,126]);box on;



% 4 Hz
subplot(3,10,[1,1.15]+20)
imagesc(0.15:0.1:0.55, 1:length(data3) ,data3, [1 2])
yticks([])
xticks([0.15:0.1:0.55])
xticklabels({'0.15','','','','0.55'})
ylabel('participants')
xlabel('\sigma')
cb = colorbar('Ticks',[1,2], 'TickLabels',{'music','speech'},'location','westoutside');
cb.Ruler.TickLabelRotation=90;
cb.Label.String = 'response';

subplot(3,10,[2.5,4]+20)
p = plot(0.15:0.1:0.55,fittedLine_3,'color',[col(3,:),0.35], 'LineWidth', 1);
hold on
p = plot(0.15:0.1:0.55,mean(fittedLine_3),'color','k', 'LineWidth', 2);
xticks(0.15:0.1:0.55)
xlim([0.15,0.55])
xtickangle(45)
xlabel('\sigma')
yticks([1,2])
yticklabels({'music','speech'})
ylim([1,2])
ylabel('response')
ytickangle(90)
set(gca,'fontsize',12)
title('4 Hz')
% grid on
box on

subplot(3,10,[9,10]+20)
scatter(slope_3,dataAll.gen,100,'filled','MarkerFaceColor',col(3,:),'MarkerFaceAlpha',.7);xlabel('response slope');title('4 Hz');set(gca,'fontsize',14);ylim([18,126]);box on;





subplot(1,10,[5.5,7.5])

bar(1,mean(slope_1),'facecolor',col(1,:),'LineWidth',2);hold on
bar(2,mean(slope_2),'facecolor',col(2,:),'LineWidth',2);
bar(3,mean(slope_3),'facecolor',col(3,:),'LineWidth',2);
er = errorbar([1,2,3],[mean(slope_1),mean(slope_2),mean(slope_3)],...
    [std(slope_1),std(slope_2),std(slope_3)]/sqrt(length(slope_3)),'LineWidth',2,'CapSize',20);   
er.Color = [0 0 0];                            
er.LineStyle = 'none'; 
scatter(ones(length(slope_1),1),slope_1,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
scatter(ones(length(slope_2),1)*2,slope_2,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
scatter(ones(length(slope_3),1)*3,slope_3,'jitter','on','MarkerEdgeColor',[0.2,0.2,0.2],'LineWidth',1, 'MarkerEdgeAlpha',.7);
xticks([1,2,3])
xlim([0.25,3.75])
xticklabels({'1 Hz','2.5 Hz','4 Hz'})
xlabel('peak AM frequency')
ylabel('response slope (regression coefficient)')
set(gca,'fontsize',14)



%% estimate required N

sampsizepwr('t',[mean(slope_1),std(slope_1)],0,0.8)
sampsizepwr('t',[mean(slope_2),std(slope_2)],0,0.8)
sampsizepwr('t',[mean(slope_3),std(slope_3)],0,0.8)


%% fit logistic function


a1 = [];
b1 = [];
r2_logistic1 = [];
for n = 1:size(data1,1)
    [xData, yData] = prepareCurveData( 0.15:0.1:0.55, data1(n,:)-1 );
    
    % Set up fittype and options.
    ft = fittype( '1/(1+exp(-b*(x-a)))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Robust = 'LAR';
    opts.StartPoint = [0.35 0.5];
    opts.Lower = [0.15 -Inf];
    opts.Upper = [0.55 Inf];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a1(n) = fitresult.a;
    b1(n) = fitresult.b;
    r2_logistic1(n) = gof.rsquare;


end


a2 = [];
b2 = [];
r2_logistic2 = [];
for n = 1:size(data2,1)
    [xData, yData] = prepareCurveData( 0.15:0.1:0.55, data2(n,:)-1 );
    
    % Set up fittype and options.
    ft = fittype( '1/(1+exp(-b*(x-a)))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Robust = 'LAR';
    opts.StartPoint = [0.35 0.5];
    opts.Lower = [0.15 -Inf];
    opts.Upper = [0.55 Inf];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a2(n) = fitresult.a;
    b2(n) = fitresult.b;
    r2_logistic2(n) = gof.rsquare;


end


a3 = [];
b3 = [];
r2_logistic3 = [];
for n = 1:size(data3,1)
    [xData, yData] = prepareCurveData( 0.15:0.1:0.55, data3(n,:)-1 );
    
    % Set up fittype and options.
    ft = fittype( '1/(1+exp(-b*(x-a)))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Robust = 'LAR';
    opts.StartPoint = [0.35 0.5];
    opts.Lower = [0.15 -Inf];
    opts.Upper = [0.55 Inf];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    a3(n) = fitresult.a;
    b3(n) = fitresult.b;
    r2_logistic3(n) = gof.rsquare;


end

[~,p, ~, stat] = ttest(b1)
[~,p, ~, stat] = ttest(b2)
[~,p, ~, stat] = ttest(b3)

[~,p, ~, stat] = ttest([r2_logistic1, r2_logistic2, r2_logistic3], [r2_1,r2_2,r2_3])

mean([r2_logistic1, r2_logistic2, r2_logistic3] - [r2_1,r2_2,r2_3])
median([r2_logistic1, r2_logistic2, r2_logistic3] - [r2_1,r2_2,r2_3])
mean([r2_logistic1, r2_logistic2, r2_logistic3]<0)

