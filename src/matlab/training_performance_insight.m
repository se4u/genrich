function training_performance_insight(a, tpe)
% USAGE
% a contains evaluations done while training.
% So for example when I am training I might say that I want to check my
% performance on the training sample itself after every (25) training
% samples. This values of (25) is (tpe) train_per_eval
if nargin < 2
    tpe=25;
end
figure();
b=smooth(a);
subplot(2,1,1); plot(tpe*(1:length(a)), a);
subplot(2,1,2); plot(tpe*(1:length(b)), b, 'r'); 
% Apparently doing a get gca immediately afterwards is error prone so I
% need to pause a little
pause(1);
xtk=get(gca, 'Xtick');
set(gca, 'XtickLabel', sprintf('%.0f|', xtk));
get_mean_std = @(x) [std(x) mean(x)];
hold on; 
for i=2:length(xtk)-1
    x_ = xtk(i-1)/tpe+1:xtk(i)/tpe;
    sm=get_mean_std(b(x_));
    s=sm(1);
    m=sm(2);
    plot(x_*tpe, ones(size(x_))*(m+s), 'k');
    plot(x_*tpe, ones(size(x_))*(m-s), 'k');
end
grid minor;
legend('5 smoothed', 'Location', 'southeast');
xlabel('The number of examples trained on so far');
ylabel('Estimate of Training accuracy');
title('No mini batch, SGD step size 1');
filename=sprintf('training_performance_%0.0f.png',1000*rand(1));
saveas(gcf(), filename);
disp(['Plot saved as ' filename]);
end