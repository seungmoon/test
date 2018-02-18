%==========================================================================
%
% Solving Huggett (93) using EGM
%
% Advanced Macro Theory PS set #5 q.4-(c),(d), q.5,6
%
%==========================================================================

clc
clear
close all

%% 1. Define parameters

% Economic Parameters
par.phi   = -2;    % Borrowing limit
par.gamma = 2;     % Coeffcient of relative risk aversion
par.beta  = 0.99;  % Discount factor
par.wage = 1;      % wage

% Numerical parameters
mpar.na   = 10;   % Number of points on the asset grid
mpar.ns   = 5;     % Number of points on the income grid
mpar.crit = 1e-5; % Numerical precision

init = 5;
rmax = 1/par.beta-1 ; % max is complete market r
rmin = -0.017;        % arbitrary (lowerbound)
amin = par.phi;       % borrowing const
amax = 10;            % upper bound

%% 2. Generate Grids

grid.a = linspace(par.phi,amax,mpar.na);
grid.s = [0.6177 0.8327 1.0000 1.2009 1.6188]; % earning shock

P  = [0.7497  0.2161 0.0322 0.002  0      ; 
      0.2161  0.4708 0.2569 0.0542 0.002  ; 
      0.0322  0.2569 0.4218 0.2569 0.0322 ; 
      0.002   0.0542 0.2569 0.4708 0.2161 ; 
      0       0.002  0.0322 0.2161 0.7497 ];    % Transition Mx


%% 3. find equilibrium asset level (bisection)

tic
r = (rmax+rmin)/2; 

%% EGM
count=0;
%while abs(init) > mpar.crit
while count < 1
     [mesh.s, mesh.a] = ndgrid(grid.s,grid.a); 
      C     = par.wage*mesh.s + r*mesh.a;
      mutil = @(c)(1./c.^(par.gamma)); 
      invmutil = @(mu)(1./mu.^(1/par.gamma));  

      Cold = C; 
      aprime = mesh.a; 
      dist = 9999;    
      while dist>mpar.crit
        mu  = mutil(C); 
        emu = P*mu;     
        Cstar = invmutil(par.beta *(1+r) * emu);  
        Astar = (Cstar  + mesh.a - par.wage.*mesh.s)/(1+r);
        for s=1:mpar.ns
            Savings=griddedInterpolant(Astar(s,:),grid.a);  
            Aprime(s,:)=Savings(grid.a); 
            Consumption=griddedInterpolant(Astar(s,:),Cstar(s,:)); 
            C(s,:)=Consumption(grid.a);             
        end
        BorrowingConstrained= mesh.a<repmat(Astar(:,1),1,mpar.na); 
        C(BorrowingConstrained)=(1+r).*mesh.a(BorrowingConstrained) +par.wage.*mesh.s(BorrowingConstrained) -par.phi; 
        dist=max(abs(C(:)-Cold(:)));  
        Cold=C;
      end
       Aprime(BorrowingConstrained)=par.phi; 
       
     %% (c). Construct transition matrix for assets and incomes (sparse)
     
       [~,idx]=histc(Aprime,grid.a);    
       idx(Aprime<=grid.a(1))=1;       
       idx(Aprime>=grid.a(end))=mpar.na-1; 
       distance = Aprime - grid.a(idx); 
       weightright=min(distance./(grid.a(idx+1)-grid.a(idx)),1); 
       weightleft=1-weightright;
       [ind1, ind2] = ndgrid(1:mpar.ns,1:mpar.na); 
       row = sub2ind([mpar.ns,mpar.na],ind1(:),ind2(:));  
       rowindex=[]; colindex=[]; value=[]; 
       for s = 1:mpar.ns 
           pi = repmat(P(:,s),mpar.na,1); 
           rowindex=[rowindex; row]; 
           col = sub2ind([mpar.ns,mpar.na],s*ones(mpar.ns*mpar.na,1),idx(:)); 
           colindex=[colindex;col];  
           rowindex=[rowindex; row]; 
           col = sub2ind([mpar.ns,mpar.na],s*ones(mpar.ns*mpar.na,1),idx(:)+1); 
           colindex=[colindex;col];  
           value=[value; pi.*weightleft(:); pi.*weightright(:)]; 
       end
       Transition = sparse(rowindex, colindex, value, mpar.ns*mpar.na, mpar.ns*mpar.na); 
       
       %% (d). Calculate the unit-eigenvector of the transition matrix
       
        [distr,eigen]=eigs(Transition',1);  
        distr=distr./sum(distr);
        distr= reshape(distr, mpar.ns,mpar.na);

       %% (e). Calculate excess Demand for assets

        ExcessA = Aprime(:)'*distr(:);
disp(ExcessA);
       %% (f). Use Bisection to update r

            if ExcessA > 0
                rmax = (r+rmax)/2;
            else
                rmin = (r+rmin)/2;
            end
            init = rmax-rmin;
            disp('Starting Iteration for r. Difference remaining:                      ');
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b %20.9f \n',init); 
           % r = (rmax+rmin)/2;
disp(rmax);
disp(rmin);
count=count+1;
end

toc