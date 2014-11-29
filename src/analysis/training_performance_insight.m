function training_performance_insight(a)
figure();
b=smooth(a);
%c=smooth(b);
subplot(2,1,1); plot(25*(1:length(a)), a);
subplot(2,1,2); plot(25*(1:length(b)), b, 'r'); 
% Apparently doing a get gca immediately afterwards is error prone so I
% need to pause a little
pause(1);
xtk=get(gca, 'Xtick');
set(gca, 'XtickLabel', sprintf('%.0f|', xtk));
get_mean_std = @(x) [std(x) mean(x)];
hold on; 
for i=2:length(xtk)-1
    x_ = xtk(i-1)/25+1:xtk(i)/25;
    sm=get_mean_std(b(x_));
    s=sm(1);
    m=sm(2);
    plot(x_*25, ones(size(x_))*(m+s), 'k');
    plot(x_*25, ones(size(x_))*(m-s), '');
end
grid minor;
legend('5 smooth', '25 smooth', 'Location', 'southeast');
xlabel('The number of examples trained on so far');
ylabel('Estimate of Training accuracy');
title('No mini batch, SGD step size 1');
filename=sprintf('training_performance_%0.0f.png',1000*rand(1));
saveas(gcf(), filename);
disp(['Plot saved as ' filename]);
