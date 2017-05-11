close all;

sigma = eye(2,2);
mu = [2, 3];
r = mvnrnd(mu, sigma, 10);
m1 = size(r, 1);
figure  
hold on
for c = 1:m1
  ncr = mvnrnd([r(c,1), r(c,2)], sigma/5, 10);
  plot(ncr(:,1), ncr(:,2), '*')
  text(r(c,1), r(c,2), num2str(c))
end
legend(strread(num2str(1:m1),'%s'))
plot(r(:,1), r(:,2), '+')

hold off
