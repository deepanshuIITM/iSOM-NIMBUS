clc
clear all
close all

%% DTLZ2
Tb = readtable('dtlz7_nsga3.csv');  
d = table2array(Tb); data = d(2:end,2:14);
F(:,1) = data(:,1); F(:,2) = data(:,2); F(:,3) = data(:,3);
dataf = data(data(:,1)>0.1554218 & data(:,2)>0.1554228 & data(:,3)<5.380035 ,:);
Tb1 = readtable('dtlz7_alter.xlsx');
d1 = table2array(Tb1);
alter = reshape(d1,[3,17])';
start_p1 = [0.1554218 0.1554228 5.380035]; 
post_cls = [0.2512642 0.2508806 5.146735; 0.8552533 0.8552533 2.616506;...
    0.7173886 0.7173908 3.901683; 0.6679063 0.6679096 4.645172];

%% Plot the solution points.
figure(1)
scatter3(F(:,1),F(:,2),F(:,3),3,'ko','filled'); 
hold on;
scatter3(dataf(:,1),dataf(:,2),dataf(:,3),8,'bo','filled');
scatter3(start_p1(:,1),start_p1(:,2),start_p1(:,3),40,'dk','filled');
scatter3(alter(:,1),alter(:,2),alter(:,3),10,'go','filled');
scatter3(post_cls(:,1),post_cls(:,2),post_cls(:,3),20,'ms','filled');
scatter3(alter(12,1),alter(12,2),alter(12,3),20,'ro','filled');
legend('Pareto Front','Post Class','Start Point','Inter Soln','Alter Soln',...
    'fin soln','Location','best')
xlabel('F1')
ylabel('F2')
zlabel('F3')

%%
f1=max(F(:,1))-min(F(:,1));
f2=max(F(:,2))-min(F(:,2));
f3=max(F(:,3))-min(F(:,3));

sum = f1+f2+f3;
F_final = (f1/sum)*F(:,1)+(f2/sum)*F(:,2)+(f3/sum)*F(:,3);

%% Making DataStruct for SOM Training
datar = data(:,4:end);
test =[datar F_final];
test1=[datar F(:,1)];
test2=[datar F(:,2)];
test3=[datar F(:,3)];
test_F = [datar F(:,1) F(:,2) F(:,3)];

%%
sData_F = som_data_struct(test_F,'my-data','comp_names',{'x1','x2','x3','x4',...
    'x5','x6','x7','x8','x9','x10','F1','F2','F3',});
sData_F = som_normalize(sData_F,'range');

sData = som_data_struct(test,'my-data','comp_names',{'x1','x2','x3','x4',...
    'x5','x6','x7','x8','x9','x10','F'});
sData = som_normalize(sData,'range');

sData1 = som_data_struct(test1,'my-data','comp_names',{'x1','x2','x3','x4',...
    'x5','x6','x7','x8','x9','x10','F1'});
sData1 = som_normalize(sData1,'range');

sData2 = som_data_struct(test2,'my-data','comp_names',{'x1','x2','x3','x4',...
    'x5','x6','x7','x8','x9','x10','F2'});
sData2 = som_normalize(sData2,'range');

sData3 = som_data_struct(test3,'my-data','comp_names',{'x1','x2','x3','x4',...
    'x5','x6','x7','x8','x9','x10','F3'});
sData3 = som_normalize(sData3,'range');

%% Initializing SOM Map Codebook Vectors (Linear Initialization)
[sMap_F]= som_lininit(sData_F,'lattice','hexa','msize',[35,35]);
[sMap]= modifiedsom_lininit(sData,'lattice','hexa','msize',[35,35]);
sMap.codebook(:,8) = sMap.codebook(:,8)*0; % optional, it does not effect the results
[sMap1]= sMap;
[sMap2]= sMap;
[sMap3]= sMap;

%% Training SOM
[sMap_F,sTrainF] = som_batchtrain(sMap_F,sData_F,'sample_order','ordered','trainlen',200);
[sMap1,sTrain1] = modifiedsom_batchtrain(sMap1,sData1,'sample_order','ordered','trainlen',200);
[sMap2,sTrain2] = modifiedsom_batchtrain(sMap2,sData2,'sample_order','ordered','trainlen',200);
[sMap3,sTrain3] = modifiedsom_batchtrain(sMap3,sData3,'sample_order','ordered','trainlen',200);

%% Denormalizing the data
sMap_F = som_denormalize(sMap_F,sData_F);
sData_F = som_denormalize(sData_F,'remove');

sMap1=som_denormalize(sMap1,sData1);
sData1=som_denormalize(sData1,'remove');

sMap2=som_denormalize(sMap2,sData2);
sData2=som_denormalize(sData2,'remove');

sMap3 = som_denormalize(sMap3,sData3);
sData3 = som_denormalize(sData3,'remove');

%%
sMap_umatrix = sMap_F;
sMap_umatrix.codebook(:,1:11) = sMap1.codebook(:,1:11);
sMap_umatrix.codebook(:,12) = sMap2.codebook(:,11);
sMap_umatrix.codebook(:,13) = sMap3.codebook(:,11);

%% Visualization of SOM results( U Matrix and Component Planes )
figure(2) 
som_show(sMap_F,'bar','none');

figure(3) 
som_show(sMap_umatrix,'umat',11:13,'comp',11:13,'bar','none');

%% iSOM Grid in function space  
figure(4)
som_grid(sMap_umatrix,'coord',sMap_umatrix.codebook(:,11:13),'label',sMap_umatrix.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k'...
    ,'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(F(:,1),F(:,2),F(:,3),20,'ro','filled');
xlabel('F1')
ylabel('F2')
zlabel('F3')

%% cSOM Grid in function space 
figure(5)
som_grid(sMap_F,'coord',sMap_F.codebook(:,11:13),'label',sMap_F.labels,'labelcolor','c','labelsize',10, 'marker','o','MarkerColor','k'...
    ,'MarkerSize',7,'linecolor', 'k');
hold on, scatter3(F(:,1),F(:,2),F(:,3),20,'ro','filled');
xlabel('F1')
ylabel('F2')
zlabel('F3')

%% BMU feasible
map_s = [35,35];
BMUs3 = som_bmus41(sMap_umatrix.codebook(:,11:end), alter);
h2 = zeros(map_s(1)*map_s(2),1);
j = 1;
for i = 1:size(BMUs3(:,1),1)
    n = BMUs3(i);
    h2(n,1)= h2(n,1)+1;
end

BMUs_sol = som_bmus41(sMap_umatrix.codebook(:,11:end), alter(9,:));
h_sol = zeros(map_s(1)*map_s(2),1);
j= 1;
for i = 1:size(BMUs_sol(:,1),1)
    n = BMUs_sol(i);
    h_sol(n,1)= h_sol(n,1)+1;
end

%% BMU for starting point
BMUs1 = som_bmus41(sMap_umatrix.codebook(:,11:end), start_p1);
hit1 = zeros(map_s(1)*map_s(2),1);

for i = 1:size(BMUs1,1)
    n = BMUs1(i);
    hit1(n,1)= 2;
end

%% BMU for alternative solution
BMUs2 = som_bmus41(sMap_umatrix.codebook(:,11:end), post_cls);
hit2 = zeros(map_s(1)*map_s(2),1);

for i = 1:size(BMUs2,1)
    n = BMUs2(i);
    hit2(n,1)= 1;
end

%% Umat Based
dmat = som_normalize(som_dmat(sMap_umatrix.codebook(:,11:12)),'range');
h1 = ones(map_s(1)*map_s(2),1);
for i = 1:size(h1,1)
    if dmat(i)<0.35
        h1(i,1) = 0;
    end
end

%% BMU for feasible region
h_fes1=zeros(map_s(1)*map_s(2),1); h_fes2=zeros(map_s(1)*map_s(2),1); h_fes3=zeros(map_s(1)*map_s(2),1);

h_fes1(find(sMap_umatrix.codebook(:,11)<start_p1(1)))=1;
h_fes2(find(sMap_umatrix.codebook(:,12)<start_p1(2)))=1;
h_fes3(find(sMap_umatrix.codebook(:,13)>start_p1(3)))=1;

%% BMU for feasible region
h_fes=zeros(map_s(1)*map_s(2),1); 
h_fes(find(sMap_umatrix.codebook(:,11)>start_p1(1) & sMap_umatrix.codebook(:,12)>start_p1(2)...
    & sMap_umatrix.codebook(:,13)<start_p1(3)))=1;

%% Plot
figure(8)
som_show(sMap_umatrix,'umat',11:12,'comp',11:13);
% som_show(sMap2,'umat',1:4,'comp',4,'bar','none');
som_show_add('hit',h_fes1,'Markersize',1,'MarkerColor','none','EdgeColor','r','Subplot',2);
som_show_add('hit',h_fes2,'Markersize',1,'MarkerColor','none','EdgeColor','r','Subplot',3);
som_show_add('hit',h_fes3,'Markersize',1,'MarkerColor','none','EdgeColor','r','Subplot',4);
som_show_add('hit',h_fes,'Markersize',1,'MarkerColor','none','EdgeColor','w','Subplot',2:4);
som_show_add('hit',h1,'Markersize',1,'MarkerColor','none','EdgeColor','k','Subplot',2:4);
som_show_add('hit',h2,'Markersize',1,'MarkerColor','g','EdgeColor','g','Subplot',2:4);
som_show_add('hit',hit1,'Markersize',1,'MarkerColor','k','EdgeColor','k','Subplot',2:4);
som_show_add('hit',hit2,'Markersize',1,'MarkerColor','m','EdgeColor','m','Subplot',2:4);
som_show_add('hit',h_sol,'Markersize',1,'MarkerColor','r','EdgeColor','r','Subplot',2:4);

% figure(9)
% som_show(sMap_umatrix,'empty','Labels');
% som_show_add('label',sMlabel.labels,'textsize',8,'textcolor','r','Subplot',1);

%% cSOM+iSOM training
% [sMap1,sTrain1] = som_batchtrain(sMap1,sData1,'sample_order','ordered','trainlen',200);
% [sMap2,sTrain2] = som_batchtrain(sMap2,sData2,'sample_order','ordered','trainlen',200);
% [sMap3,sTrain3] = som_batchtrain(sMap3,sData3,'sample_order','ordered','trainlen',200);

%%
% figure(10)
% parallelcoords(F,'Labels',{'F1','F2','F3'},'LineWidth',2,'color','y'); hold on;
% parallelcoords(start_p1,'Labels',{'F1','F2','F3'},'LineWidth',2,'color','b')
% parallelcoords(alter,'Labels',{'F1','F2','F3'},'LineWidth',1,'color','g')
% parallelcoords(post_cls,'Labels',{'F1','F2','F3'},'LineWidth',1.5,'color','m')
% parallelcoords(alter(9,:),'Labels',{'F1','F2','F3'},'LineWidth',2,'color','k')
