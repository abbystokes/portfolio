
% Perceptron function 


function [w] = perceptron(X,Y,w_init,rate)

w = w_init;
N = size(X,2);
pos_idx = (Y==1);
neg_idx = (Y==-1);
mxx = max(X(1,:));
mnx=-0.1*mxx;%for visualization %mnx = min(X(1,:));
mxy = max(X(2,:));
mny=-0.1*mxy; % for visualization %mny = min(X(2,:));

figure;
ginput(1);
err = 1;
round = 0;
while err > 0  
  for ii = 1 : N         %cycle through training set
    if sign(w'*X(:,ii)) ~= Y(ii) %wrong decision?
        w = w + rate*X(:,ii) * Y(ii);   %then add (or subtract) this point to w
        x1=mnx:0.01:mxx;
        x2=-(w(1)*x1+w(3))/w(2);
        %figure;
        clf;
        hold on
        plot(X(1,pos_idx),X(2,pos_idx),'b*','MarkerSize',10);
        plot(X(1,neg_idx),X(2,neg_idx),'r+','MarkerSize',10);
        plot(X(1,ii),X(2,ii),'ko','MarkerSize',15);
        plot(x1,x2);
        xlim([mnx mxx]);
        ylim([mny mxy]);
        %ginput(1);
        pause(0.5); %change the delay
    end
  end
  round = round + 1
  err = sum(sign(w'*X)~=Y')/N   %show misclassification rate
end



% Perceptron function 


function [w] = perceptron(X,Y,w_init,rate)

w = w_init;
N = size(X,2);
pos_idx = (Y==1);
neg_idx = (Y==-1);
mxx = max(X(1,:));
mnx=-0.1*mxx;%for visualization %mnx = min(X(1,:));
mxy = max(X(2,:));
mny=-0.1*mxy; % for visualization %mny = min(X(2,:));

figure;
ginput(1);
err = 1;
round = 0;
while err > 0  
  for ii = 1 : N         %cycle through training set
    if sign(w'*X(:,ii)) ~= Y(ii) %wrong decision?
        w = w + rate*X(:,ii) * Y(ii);   %then add (or subtract) this point to w
        x1=mnx:0.01:mxx;
        x2=-(w(1)*x1+w(3))/w(2);
        %figure;
        clf;
        hold on
        plot(X(1,pos_idx),X(2,pos_idx),'b*','MarkerSize',10);
        plot(X(1,neg_idx),X(2,neg_idx),'r+','MarkerSize',10);
        plot(X(1,ii),X(2,ii),'ko','MarkerSize',15);
        plot(x1,x2);
        xlim([mnx mxx]);
        ylim([mny mxy]);
        %ginput(1);
        pause(0.5); %change the delay
    end
  end
  round = round + 1
  err = sum(sign(w'*X)~=Y')/N   %show misclassification rate
end


% Run perceptron on random data points in two classes 

n = 100; %set the number of data points
mydata = rand(n,2);

shiftidx = abs(mydata(:,1)-mydata(:,2))>0.05;
mydata = mydata(shiftidx,:); 
myclasses = mydata(:,1)>mydata(:,2); % labels
n = size(mydata,1);
X = [mydata ones(1,n)']'; Y=myclasses;
Y = Y * 2 -1; 

% init weigth vector
w = [mean(mydata) 0]';

for i = 1:1
    w=rand(1,3)';
    w(3,1)=0;%go through the origin for visualization
    % call perceptron
    wtag=perceptron(X,Y,w,10);
end

% call perceptron
% wtag=perceptron(X,Y,w);
% predict
ytag=wtag'*X;

% plot prediction over origianl data

%plot(X(1,ytag<0),X(2,ytag<0),'bo')
%plot(X(1,ytag>0),X(2,ytag>0),'ro')
%legend('class -1','class +1','pred -1','pred +1')