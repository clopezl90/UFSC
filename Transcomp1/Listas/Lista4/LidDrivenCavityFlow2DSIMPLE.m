% Kevin Viveros
% SIMPLE Algorithm Code for 2D Lid-Driven Cavity Flow
% MAE 571: Computational Fluid Dynamics

clear all; close all; clc;

%% Generate Mesh %%
L = 1;                                     % Length of cavity
H = 1;                                     % Height of cavity
nx = 41; Imax = nx; imax = Imax-1;         % Number of nodes in x
ny = nx; Jmax = ny; jmax = Jmax-1;         % Number of nodes in y
dx = L / (nx-1);                           % Width of space step in x
dy = H / (ny-1);                           % Width of space step in y
x = 0 : dx : L; y = H : -dy : 0;           % Range of y (Pressure Grid)
ux = dx/2 : dx : L-dx/2; uy = H : -dy : 0; % Range of x (u vel Grid)
vx = 0 : dx : L; vy = H-dy/2 : -dy : dy/2; % Range of x (v vel Grid)

%% Guess Pressure and Velocity Fields %%

% Preallocating u %
u_star = zeros(ny,nx-1);             
u_star2 = zeros(size(u_star));
u_prime = zeros(size(u_star));
Xewns2 = zeros(size(u_star));

% Preallocating v %
v_star = zeros(ny-1,nx);
v_star2 = zeros(size(v_star));
v_prime = zeros(size(v_star));
Yewns2 = zeros(size(v_star));

% Preallocating p %
p_star = zeros(ny,nx);
p_prime = zeros(size(p_star));
p_prime2 = zeros(size(p_star));
pepsil = 100
% Flow Parameters %
lidvel = 1; % m/s
Re = 1e-3; % Flow Reynolds number

% Discretization Parameters %
alpha_p = 0.08; % Under Relaxation Factor
alpha_u = 0.05; % Under Relaxation Factor
alpha_v = 0.05; % Under Relaxation Factor

% Boundary Conditions %
u_star(1,:) = lidvel; u_star(end,:) = 0; u_star2 = u_star;
v_star(:,1) = 0; v_star(:,end) = 0; v_star2 = v_star;

%% SIMPLE Algorithm Loop %%

tol = 1e-7
maxIter = 3000
IterNum = 0;
for Iter = 1 : maxIter
    IterNum = IterNum + 1
    
%% X-Momentum %%

uerr = 100;
for II = 1 : maxIter
    for i = 1 : imax; I = i;
        for J = 2 : Jmax-1; j = J;
            % North Face u_star1
            Xnorth = (((v_star(j-1,I)+v_star(j-1,I+1))/4)-((1/Re)*(1/dy)))*dx;
            % South Face u_star1
            Xsouth = (((v_star(j,I)+v_star(j,I+1))/4)+((1/Re)*(1/dy)))*dx;
            
            % North Face u_star2
            Xnorth2 = (((v_star(j-1,I)+v_star(j-1,I+1))/4)+((1/Re)*(1/dy)))*dx;
            % South Face u_star2
            Xsouth2 = (((v_star(j,I)+v_star(j,I+1))/4)-((1/Re)*(1/dy)))*dx;
            % North Face u_star2 prime
            Xnorth2_prime = (((v_prime(j-1,I)+v_prime(j-1,I+1))/4)+((1/Re)*(1/dy)))*dx;
            % South Face u_star2 prime
            Xsouth2_prime = (((v_prime(j,I)+v_prime(j,I+1))/4)-((1/Re)*(1/dy)))*dx;
            
            if i == 1 % Left Wall
%                 % West Face u_star1
%                 Xwest = (((0+u_star(J,i))/4)+((1/Re)*(2/dx)))*dy;
                % East Face u_star1
                Xeast = (((u_star(J,i+1)+u_star(J,i))/4)-((1/Re)*(1/dx)))*dy;
                
                % West Face u_star2
                Xwest2 = (((0+u_star(J,i))/4)-((1/Re)*(2/dx)))*dy;
                % East Face u_star2
                Xeast2 = (((u_star(J,i+1)+u_star(J,i))/4)+((1/Re)*(1/dx)))*dy;
                % West Face u_star2 prime
                Xwest2_prime = (((0+u_prime(J,i))/4)-((1/Re)*(2/dx)))*dy;
                % East Face u_star2 prime
                Xeast2_prime = (((u_prime(J,i+1)+u_prime(J,i))/4)+((1/Re)*(1/dx)))*dy;
                
                Xewns2(J,i) = Xeast2-Xwest2+Xnorth2-Xsouth2;
                Xewns2_prime(J,i) = Xeast2_prime-Xwest2_prime+Xnorth2_prime-Xsouth2_prime;
                
                u_star2(J,i) = (-((p_star(J,I+1)-p_star(J,I))/dx)*dx*dy...
                    -Xeast*u_star(J,i+1)...
                    -Xnorth*u_star(J-1,i)...
                    +Xsouth*u_star(J+1,i))...
                    /Xewns2(J,i);
                
            elseif i == imax % Right Wall
                % West Face u_star1
                Xwest = (((u_star(J,i-1)+u_star(J,i))/4)+((1/Re)*(1/dx)))*dy;
%                 % East Face u_star1
%                 Xeast1 = (((0+u_star(J,i))/4)-((1/Re)*(2/dx)))*dy;
                
                % West Face u_star2
                Xwest2 = (((u_star(J,i-1)+u_star(J,i))/4)-((1/Re)*(1/dx)))*dy;
                % East Face u_star2
                Xeast2 = (((0+u_star(J,i))/4)+((1/Re)*(2/dx)))*dy;
                % West Face u_star2 prime
                Xwest2_prime = (((u_prime(J,i-1)+u_prime(J,i))/4)-((1/Re)*(1/dx)))*dy;
                % East Face u_star2 prime
                Xeast2_prime = (((0+u_prime(J,i))/4)+((1/Re)*(2/dx)))*dy;
                
                Xewns2(J,i) = Xeast2-Xwest2+Xnorth2-Xsouth2;
                Xewns2_prime(J,i) = Xeast2_prime-Xwest2_prime+Xnorth2_prime-Xsouth2_prime;
                
                u_star2(J,i) = (-((p_star(J,I+1)-p_star(J,I))/dx)*dx*dy...
                    +Xwest*u_star(J,i-1)...
                    -Xnorth*u_star(J-1,i)...
                    +Xsouth*u_star(J+1,i))...
                    /Xewns2(J,i);
                
            else % Interior
                % West Face u_star1
                Xwest = (((u_star(J,i-1)+u_star(J,i))/4)+((1/Re)*(1/dx)))*dy;
                % East Face u_star1
                Xeast = (((u_star(J,i+1)+u_star(J,i))/4)-((1/Re)*(1/dx)))*dy;
                
                % West Face u_star2
                Xwest2 = (((u_star(J,i-1)+u_star(J,i))/4)-((1/Re)*(1/dx)))*dy;
                % East Face u_star2
                Xeast2 = (((u_star(J,i+1)+u_star(J,i))/4)+((1/Re)*(1/dx)))*dy;
                
                % West Face u_star2 prime
                Xwest2_prime = (((u_prime(J,i-1)+u_prime(J,i))/4)-((1/Re)*(1/dx)))*dy;
                % East Face u_star2 prime
                Xeast2_prime = (((u_prime(J,i+1)+u_prime(J,i))/4)+((1/Re)*(1/dx)))*dy;
                
                Xewns2(J,i) = Xeast2-Xwest2+Xnorth2-Xsouth2;
                Xewns2_prime(J,i) = Xeast2_prime-Xwest2_prime+Xnorth2_prime-Xsouth2_prime;
                
                u_star2(J,i) = (-((p_star(J,I+1)-p_star(J,I))/dx)*dx*dy...
                    -Xeast*u_star(J,i+1)...
                    +Xwest*u_star(J,i-1)...
                    -Xnorth*u_star(J-1,i)...
                    +Xsouth*u_star(J+1,i))...
                    /Xewns2(J,i);

            end
        end
    end
    
    if uerr > tol
        uerr = max(max(abs(u_star2-u_star)));
    u_star = u_star2;
    else
        break
    end
end

% pause
%% Y-Momentum %%

verr = 100;
for II = 1 : maxIter
    for j = 1 : jmax; J = j;
        for I = 2 : Imax-1; i = I;
            % West Face v_star1
            Ywest = (((u_star2(J+1,i-1)+u_star2(J,i-1))/4)+((1/Re)*(1/dx)))*dy;
            % East Face v_star1
            Yeast = (((u_star2(J+1,i)+u_star2(J,i))/4)-((1/Re)*(1/dx)))*dy;
            
            % West Face v_star2
            Ywest2 = (((u_star2(J+1,i-1)+u_star2(J,i-1))/4)-((1/Re)*(1/dx)))*dy;
            % East Face v_star2
            Yeast2 = (((u_star2(J+1,i)+u_star2(J,i))/4)+((1/Re)*(1/dx)))*dy;
            
            % West Face v_star2 prime
            Ywest2_prime = (((u_prime(J+1,i-1)+u_prime(J,i-1))/4)-((1/Re)*(1/dx)))*dy;
            % East Face v_star2 prime
            Yeast2_prime = (((u_prime(J+1,i)+u_prime(J,i))/4)+((1/Re)*(1/dx)))*dy;
            
            if j == 1 % Top Wall
%                 % North Face v_star1
%                 Ynorth = (((0+v_star(j,I))/4)-((1/Re)*(2/dy)))*dx;
                % South Face v_star1
                Ysouth = (((v_star(j+1,I)+v_star(j,I))/4)+((1/Re)*(1/dy)))*dx;
                
                % North Face v_star2
                Ynorth2 = (((0+v_star(j,I))/4)+((1/Re)*(2/dy)))*dx;
                % South Face v_star2
                Ysouth2 = (((v_star(j+1,I)+v_star(j,I))/4)-((1/Re)*(1/dy)))*dx;
                
                % North Face v_star2 prime
                Ynorth2_prime = (((0+v_prime(j,I))/4)+((1/Re)*(2/dy)))*dx;
                % South Face v_star2 prime
                Ysouth2_prime = (((v_prime(j+1,I)+v_prime(j,I))/4)-((1/Re)*(1/dy)))*dx;
                
                Yewns2(j,I) = Yeast2-Ywest2+Ynorth2-Ysouth2;
                Yewns2_prime(j,I) = Yeast2_prime-Ywest2_prime+Ynorth2_prime-Ysouth2_prime;
                
                v_star2(j,I) = (-((p_star(J,I)-p_star(J+1,I))/dy)*dy*dx...
                    -Yeast*v_star(j,I+1)...
                    +Ywest*v_star(j,I-1)...
                    +Ysouth*v_star(j+1,I))...
                    /Yewns2(j,I);
                
            elseif j == jmax % Bottom Wall
                % North Face v_star1
                Ynorth = (((v_star(j-1,I)+v_star(j,I))/4)-((1/Re)*(1/dy)))*dx;
%                 % South Face v_star1
%                 Ysouth = (((0+v_star(j,I))/4)+((1/Re)*(2/dy)))*dx;
                
                % North Face v_star2
                Ynorth2 = (((v_star(j-1,I)+v_star(j,I))/4)+((1/Re)*(1/dy)))*dx;
                % South Face v_star2
                Ysouth2 = (((0+v_star(j,I))/4)-((1/Re)*(2/dy)))*dx;
                
                % North Face v_star2 prime
                Ynorth2_prime = (((v_prime(j-1,I)+v_prime(j,I))/4)+((1/Re)*(1/dy)))*dx;
                % South Face v_star2 prime
                Ysouth2_prime = (((0+v_prime(j,I))/4)-((1/Re)*(2/dy)))*dx;
                
                Yewns2(j,I) = Yeast2-Ywest2+Ynorth2-Ysouth2;
                Yewns2_prime(j,I) = Yeast2_prime-Ywest2_prime+Ynorth2_prime-Ysouth2_prime;
                
                v_star2(j,I) = (-((p_star(J,I)-p_star(J+1,I))/dy)*dy*dx...
                    -Yeast*v_star(j,I+1)...
                    +Ywest*v_star(j,I-1)...
                    -Ynorth*v_star(j-1,I))...
                    /Yewns2(j,I);
                
            else % Interior
                % North Face v_star1
                Ynorth = (((v_star(j-1,I)+v_star(j,I))/4)-((1/Re)*(1/dy)))*dx;
                % South Face v_star1
                Ysouth = (((v_star(j+1,I)+v_star(j,I))/4)+((1/Re)*(1/dy)))*dx;
                
                % North Face v_star2
                Ynorth2 = (((v_star(j-1,I)+v_star(j,I))/4)+((1/Re)*(1/dy)))*dx;
                % South Face v_star2
                Ysouth2 = (((v_star(j+1,I)+v_star(j,I))/4)-((1/Re)*(1/dy)))*dx;
                
                % North Face v_star2 prime
                Ynorth2 = (((v_prime(j-1,I)+v_prime(j,I))/4)+((1/Re)*(1/dy)))*dx;
                % South Face v_star2 prime
                Ysouth2 = (((v_prime(j+1,I)+v_prime(j,I))/4)-((1/Re)*(1/dy)))*dx;
                
                Yewns2(j,I) = Yeast2-Ywest2+Ynorth2-Ysouth2;
                Yewns2_prime(j,I) = Yeast2_prime-Ywest2_prime+Ynorth2_prime-Ysouth2_prime;
                
                v_star2(j,I) = (-((p_star(J,I)-p_star(J+1,I))/dy)*dy*dx...
                    -Yeast*v_star(j,I+1)...
                    +Ywest*v_star(j,I-1)...
                    -Ynorth*v_star(j-1,I)...
                    +Ysouth*v_star(j+1,I))...
                    /Yewns2(j,I);
                
            end
        end
    end
    
    if verr>tol
        verr = max(max(abs(v_star2-v_star)));
    v_star = v_star2;
    else
        break
    end
end

% pause
%% Continuity (Pressure Correction)

Dx = dy./Xewns2;
Dy = dx./Yewns2;


for II = 1 : maxIter
    for I = 2 : Imax-1; i = I;
        for J = 2 : Jmax-1; j = J;
            b_prime = (u_star2(J,i-1)-u_star2(J,i))*dy + (v_star2(j,I)-v_star2(j-1,I))*dx;
            
            Cewns(J,I) = (Dx(J,i)+Dx(J,i-1))*dy + (Dy(j-1,I)+Dy(j,I))*dx;
            
            p_prime2(J,I) = ((Dx(J,i)*p_prime(J,I+1)+Dx(J,i-1)*p_prime(J,I-1))*dy...
                + (Dy(j-1,I)*p_prime(J-1,I)+Dy(j,I)*p_prime(J+1,I))*dx...
                + b_prime)...
                /Cewns(J,I);
        end
    end
        
%         Boundary Conditions
    p_prime2(1,:) = p_prime2(2,:); % Top Wall
    p_prime2(end,:) = p_prime2(end-1,:); % Bottom Wall
    p_prime2(:,1) = p_prime2(:,2); % Left Wall
    p_prime2(:,end) = p_prime2(:,end-1); % Right Wall
        
        perr = max(max(abs(p_prime2-p_prime)));
    if perr>tol
        
p_prime = p_prime2;
    else
        break
        
        
    end
end


    
%     perr = max(max(abs(p_prime2-p_prime1)));
%     p_prime1 = p_prime2;
%     if perr<tol
%         break
%     end

%     
    
    
    
p_prime2
% pause
%% Velocity Correction %%
for II = 1 : 1
% u velocity correction
for i = 1 : imax; I = i;
    for J = 2 : Jmax-1; j = J;
        u_prime(J,i) = Dx(J,i) * -(p_prime2(J,I+1)-p_prime2(J,I));
    end
end
u_prime
% pause
% v velocity correction
for j = 1 : jmax; J = j;
    for I = 2 : Imax-1; i = I;
        v_prime(j,I) = Dy(j,I) * -(p_prime2(J,I)-p_prime2(J+1,I));
    end
end
end

v_prime
% pause
%% Correct Pressure and Velocity Fields %%

% p = p_star + alpha * p_prime2
% u = u_star2 + 0.05*u_prime
% v = v_star2 + 0.05*v_prime
% % pause



% Update Pressure Field
    p_new = p_star + alpha_p*p_prime2;
    
    u = u_star2 + u_prime;
    u_new = alpha_u*u + (1-alpha_u)*u_star2;
    
    v = v_star2 + v_prime;
    v_new = alpha_v*v + (1-alpha_v)*v_star2;
    resid = max(max(abs(p_prime2)))
    if resid > tol
        p_star = p_new
        u_star = u_new
        v_star = v_new
    else
        break
    end
    
    
%% Check for Convergence %%
% 
% uepsilon = max(max(abs(u - u_star)));
% vepsilon = max(max(abs(v - v_star)));
% % pepsilon = max(max(abs(p - p_star)))
% pepsil = max(max(abs(p_prime2)))
% % pause
% p_star = p;
%     u_star = u;
%     v_star = v;
%     
%     u_star(1,:) = lidvel; u_star(end,:) = 0;
%     
% v_star(:,1) = 0; v_star(:,end) = 0;
% 
% p_star(1,:) = p_star(2,:); % Top Wall
%     p_star(end,:) = p_star(end-1,:); % Bottom Wall
%     p_star(:,1) = p_star(:,2); % Left Wall
%     p_star(:,end) = p_star(:,end-1); % Right Wall
%     
%     
%     
%     
% if pepsil < tol
%     break
% end
pause
end


%% Plots %%
[Xmesh Ymesh] = meshgrid(x,y);

[uXmesh uYmesh] = meshgrid(ux,uy);

[vXmesh vYmesh] = meshgrid(vx,vy);

figure(1)
contour(uXmesh,uYmesh,u_star)

figure(2)
contour(vXmesh,vYmesh,v_star)

figure(3)
contour(Xmesh,Ymesh,p_star)

figure(5)
streamline(Xmesh(2:end,2:end),Ymesh(2:end,2:end),u_star(2:end,:),v_star(:,2:end),Xmesh(2:end,2:end),Ymesh(2:end,2:end))
grid on

