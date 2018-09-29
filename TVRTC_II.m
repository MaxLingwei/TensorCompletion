%by Lingwei Li
%The code is a implementation for ICIP 2018 paper "TOTAL VARIATION REGULARIZED REWEIGHTED LOW-RANK TENSOR COMPLETION FOR COLOR IMAGE INPAINTING"

function [tensor_Z  ] = TVRTC_II(index, value, A)
    MODE = 2;
    Q=cell(MODE,1);
    F=cell(MODE,1);
    R=cell(MODE,1);
    U=cell(MODE,1);
    V=cell(MODE,1);
    myV=cell(MODE,1);
    
    Z=cell(MODE,1);
    W=cell(MODE,1);
    
    Lambda=cell(MODE,1);
    Gamma=cell(MODE,1);
    Phi=cell(MODE,1);

    rse_set = [];
    psnr_set = [];
    
    N = 3;
    lambda_1=0.2;
    lambda_2=1000;
    alpha=[1/MODE, 1/MODE, 1/MODE];
    beta=[1,1,0];
    sizeA = size(A);
    epsilon=1.0e-6;
    for i=1:MODE
        l=1;
        for j=[1:i-1,i+1:N]
            l=l*sizeA(j);
        end
        U{i}=rand(sizeA(i),sizeA(i));
        V{i}=rand(sizeA(i),sizeA(i));
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
    U{3}=rand(sizeA(3),sizeA(3));
    
    for i=1:MODE
       if(beta(i)==1)
           Lambda{i}=sign(Q{i})/max([norm(Q{i}), norm(Q{i},Inf), epsilon]);
           Phi{i}=sign(R{i})/max([norm(R{i}), norm(R{i},Inf), epsilon]);
       else
           Lambda{i}=[];
           Phi{i}=[];
       end    
       Gamma{i}=sign(U{i})/max([norm(U{i}), norm(U{i},Inf), epsilon]);
       
    end
    
    tensor_Z=tenzeros(sizeA);
    tensor_Z(index)=value;
    tensor_G=tensor_Z;
    tensor_W=tenzeros(sizeA);
    
    iteration=1;
    
    myInitial_v=0.2;
    rho_1=myInitial_v;
    rho_2=myInitial_v;
    rho_3=myInitial_v;
    rho_4=myInitial_v;
    
    factor=1.05;
    
    while(true)
        iteration
        for n=1:N
            Z{n}=double(tenmat(tensor_Z,n));
            W{n}=double(tenmat(tensor_W,n));
        end
        
        %update Q
        for n=1:MODE
            if(beta(n)==1)
                Q{n}=myshrinkage(F{n}*R{n} - 1/rho_1 *Lambda{n}, lambda_1/rho_1);
            end
        end
        
        %update R
        for n=1:MODE
            if(beta(n)==1)
                R{n}= (rho_1*F{n}'*F{n} + rho_2 * eye((sizeA(n))))\(F{n}'* Lambda{n}+ rho_1* F{n}'* Q{n} + rho_2 * Z{n} - Phi{n});
            end
        end
        
        %update U
        for n=1:MODE
            U{n}=SVT(V{n} + Gamma{n}/rho_3,alpha(n)/rho_3,0);
        end
        
        %update V
        for n=1:MODE
            if n == 1
                tmp=ttm(tensor_G,V{2},2);
                
            elseif n == 2
                tmp=ttm(tensor_G,V{1},1);
            end
            tmp = ttm(tmp, U{3}, 3);
            tmp=tenmat(tmp,n);
            tmp=double(tmp);
            V{n}=(- Gamma{n} + rho_3 * U{n} + W{n} * tmp' + rho_4 * Z{n}* tmp')/((rho_3*eye(sizeA(n)) + rho_4*(tmp*tmp'))); 
        end
        
        tensor_A = ttm(tensor_G, {V{1}, V{2}}, [1, 2]);
        A_3 = double(tenmat(tensor_A, 3));
        U{3} = (Z{3} + W{3} / rho_4) * A_3' / (A_3 * A_3');
        
        %update Z
        tmp=0;
        for n=1:MODE
            if(beta(n)==1)
               currZ = Phi{n} + rho_2 * R{n};
               myZ=tenmat(tensor_Z,n);
               myZ=tenmat(currZ,myZ.rdims, myZ.cdims,myZ.tsize);
               tmp=tmp + tensor(myZ);
             end
        end
        tensor_tau = ttm(tensor_G,V,1:MODE);
        tensor_tau = ttm(tensor_tau, U{3}, 3);
        tmp = tmp - tensor_W + rho_4* tensor_tau;
        
        NN=numel(find(beta==1));
        tensor_Z = tmp/(NN*rho_2 + rho_4);
        tensor_Z(index)=value;
        rse_set = [rse_set, rse(A, tensor_Z)];
        psnr_set = [psnr_set, psnr(A, tensor_Z)];
        %update G

        for n=1:MODE
            myV{n}=V{n}';
        end
        myV{3} = U{3}';
        myG = optimize_Z(myV,double(tensor_Z),double(tensor_W),rho_4,lambda_2);
        tensor_G = tensor(myG);
        
        %update multiplers
        for n=1:MODE
            if(beta(n)==1)
                Lambda{n}=Lambda{n} + rho_1 * (Q{n}- F{n}*R{n});
                Phi{n}= Phi{n}+ rho_2*(R{n} - double(tenmat(tensor_Z,n)));
            end
            Gamma{n}=Gamma{n}+ rho_3*(V{n}-U{n});
        end
        tensor_temp = ttm(tensor_G, V,1:MODE);
        tensor_temp = ttm(tensor_temp, U{3}, 3);
        tensor_W = tensor_W + rho_4*(tensor_Z - tensor_temp);
        
        rho_1=rho_1*factor;
        rho_2=rho_2*factor;
        rho_3=rho_3*factor;
        rho_4=rho_4*factor;
  
        if(iteration>200)
            break;
        end
        iteration = iteration + 1;
                
    end
    tensor_Z=double(tensor_Z);

end

