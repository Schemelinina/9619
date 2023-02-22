% Iteration code_problem 1 (redundant)
% Written by Irina Shchemelinina 

clear all
% Set parameters
beta=0.96;
alpha=1/3;
delta=1;
z=1

% make capital grid
ki=0.01;
ke=0.2;
step=0.01;
K=ki:step:ke;                 %building capital grid 
I=length(K)
%kgrid = linspace(ki,ke,20);   %building capital grid (2) - alternative

% initial guess
fk = z*K.^alpha+(1-delta)*K;    %function f(k) formula
fk=fk'*ones(1,I);             %function f(k) matrix 
v=log(fk);                    %initial guess value function assuming v0(k)=0 
for i=1:I                     %express a period return function as a function of current and future state variables
for j=1:I
C(i,j) = z*K(j).^alpha + (1-delta)*K(j) - K(i);
end
end 
U=log(C);                    %map consumptions into utilities using log function

%Iterations using initial guess for value function
T=100              
for j=1:T
w=U+beta*v;
v1=max(w);
v=v1'*ones(1,I);             %update v
end

%Building outputs
[val,ind]=max(w);            % value function is in val; future capital is in ind
optimalk = K(ind);           % optimal capital chosen by SP - policy function 
optimalc = z*K.^alpha + (1-delta)*K - optimalk;

%Plot value function
figure(1)
plot(K,v1','LineWidth',2)
xlabel('K');
ylabel('Value function');

%Plot policy function
figure(2)
plot(K,optimalk','LineWidth',2)
xlabel('K');
ylabel('Policy function');

%plot steady state convergence
figure(4)
plot(K,optimalk','LineWidth',2)
hold on
plot(K,K','-r','LineWidth',2)
xlabel('K');
ylabel('K"');

