%by Lingwei Li
%The code is a implementation for ICIP 2018 paper "TOTAL VARIATION REGULARIZED REWEIGHTED LOW-RANK TENSOR COMPLETION FOR COLOR IMAGE INPAINTING"

function [ tensor_Z ] = TVRTC_I(index, value, A)
    MODE = 2;
    N = 3;
    lambda=0.02;
    alpha=[1/N, 1/N, 1/N];
    beta=[1,1,0];
    
    M=cell(MODE,1);
    Q=cell(MODE,1);
    F=cell(MODE,1);
    Z=cell(MODE,1);
    R=cell(MODE,1);
    Lambda=cell(MODE,1);
    Gamma=cell(MODE,1);
    Phi=cell(MODE,1);
    rse_set = [];
    sizeA = size(A);
    epsilon=1.0e-6;
    
    for i=1:MODE
        l=1;
        for j=[1:i-1,i+1:N]
            l=l*sizeA(j);
        end
        M{i}=zeros(sizeA(i),l);
        if(beta(i)==1)
            Q{i}=rand(sizeA(i)-1, l);
            R{i}=rand(sizeA(i),l);
        else
            Q{i}=[];
            R{i}=[];
        end
       
        F{i}=zeros(sizeA(i)-1,sizeA(i));
        for j=1:sizeA(i)-1
            F{i}(j,j)=1;
            F{i}(j,j+1)=-1;
        end

    end
    for i=1:MODE
       if(beta(i)==1)
           Lambda{i}=sign(Q{i})/max([norm(Q{i}), norm(Q{i},Inf), epsilon]);
           Phi{i}=sign(R{i})/max([norm(R{i}), norm(R{i},Inf), epsilon]);
       else
           Lambda{i}=[];
           Phi{i}=[];
       end    
       Gamma{i}=sign(M{i})/max([norm(M{i}), norm(M{i},Inf), epsilon]);
    end
    
    tensor_Z=tenzeros(sizeA);
    tensor_Z(index)=value;
    
    iteration=1;
    myInitial_v=1.0e-3;
    rho=myInitial_v;
    mu=myInitial_v;
    gamma=myInitial_v;
    factor=1.05;
    while(true)
        iteration
        tensor_Z_pre=tensor_Z;
        for n=1:MODE
            Z{n}=double(tenmat(tensor_Z,n));
        end

        %update Q
        for n=1:MODE
            if(beta(n)==1)
                Q{n}=myshrinkage(F{n}*R{n} - 1/rho *Lambda{n}, lambda/rho);
            end
        end
        %update M
        for n=1:MODE
            tmpMatrix=Z{n}  -  1/mu*Gamma{n};
            M{n}=SVT(tmpMatrix, alpha(n)/mu,0);
        end
        %update R
        for n=1:MODE
            if(beta(n)==1)
                R{n}= (rho*F{n}'*F{n} + gamma * eye((sizeA(n))))\(F{n}'* Lambda{n}+ rho* F{n}'* Q{n} + gamma * Z{n} - Phi{n});
            end
        end
        
        
        %update Z
        tmp=0;
        for n=1:MODE
            if(beta(n)==1)
               currZ= Gamma{n} + Phi{n} +  mu * M{n} + gamma * R{n};
            else
               currZ= Gamma{n} +  mu * M{n};
            end
            myZ=tenmat(tensor_Z,n);
            myZ=tenmat(currZ,myZ.rdims, myZ.cdims,myZ.tsize);
            tmp=tmp + tensor(myZ);
        end
        NN=numel(find(beta==1));
        
        tensor_Z=tmp/(MODE*mu + NN*gamma);
        tensor_Z(index)=value;
        rse_set = [rse_set, rse(A, tensor_Z)];
        
        for n=1:MODE
            if(beta(n)==1)
                Lambda{n}=Lambda{n} + rho * (Q{n}- F{n}*R{n});
                Phi{n}= Phi{n}+ gamma*(R{n} - double(tenmat(tensor_Z,n)));
            end
            Gamma{n}=Gamma{n}+ mu*(M{n} - double(tenmat(tensor_Z,n)));
        end
        
        diff=norm(tensor_Z-tensor_Z_pre)/norm(tensor_Z);
        
        larange_cond_1=0;
        larange_cond_2=0;
        larange_cond_3=0;
        for n=1:MODE
            if(beta(n)==1)
                larange_cond_1= larange_cond_1 +  sum(sum(Lambda{n}.* (Q{n} - F{n}*R{n})));
                larange_cond_3= larange_cond_3 +  sum(sum(Phi{n}.*(R{n} - double(tenmat(tensor_Z,n)))));
            end
            larange_cond_2= larange_cond_2 +  sum(sum(Gamma{n}.* (M{n} - double(tenmat(tensor_Z,n)))));
        end

        
        rho=rho*factor;
        mu=mu*factor;
        gamma=gamma*factor;
        
        if(diff<epsilon||iteration>300)
            break;
        end
        
        iteration = iteration + 1;
        
    end
    tensor_Z=double(tensor_Z);
end

