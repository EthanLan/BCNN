function x = neighbor_process1(d,row,column,size)
%neighbor_process1 output the number of non-zero neighbor coefficients of
%central coefficient.

%input
%d: input matrix, its value should be logical.
%row and column means the demension of matrix d
%size only can be 8 or 4, which means that 8 neighbor coefficents or 4
%neighbor coefficents

%output
%x[i,j] means the number of non-zero neighbor coefficents of central
%coefficient at positon [i,j]

d = double(d);

mask0 = [1 0 0; 0 0 0; 0 0 0];
rb = conv2(d, mask0, 'same');

mask1 = [0 1 0; 0 0 0; 0 0 0];
b = conv2(d, mask1, 'same');

mask2 = [0 0 1; 0 0 0; 0 0 0];
lb = conv2(d, mask2, 'same');

mask3 = [0 0 0; 1 0 0; 0 0 0];
r = conv2(d, mask3, 'same');

mask4 = [0 0 0; 0 0 1; 0 0 0];
l = conv2(d, mask4, 'same');

mask5 = [0 0 0; 0 0 0; 1 0 0];
ru = conv2(d, mask5, 'same');

mask6 = [0 0 0; 0 0 0; 0 1 0];
u = conv2(d, mask6, 'same');

mask7 = [0 0 0; 0 0 0; 0 0 1];
lu = conv2(d, mask7, 'same');

% vector = zeros(row,column,size);
x = zeros(row,column);
for i = 1:row
    for j = 1:column
        if size == 8
            vector = [lu(i,j);u(i,j);ru(i,j);l(i,j);r(i,j);lb(i,j);b(i,j);rb(i,j)];
            if (i == 1 && j == 1),                      vector = [r(i,j);b(i,j);rb(i,j)]; end
            if (i == 1 && j == column),                 vector = [l(i,j);lb(i,j);b(i,j)]; end
            if (i == row && j == 1),                    vector = [u(i,j);ru(i,j);r(i,j)]; end
            if (i == row && j == column),               vector = [lu(i,j);u(i,j);l(i,j)]; end
            if (i == 1 && (j ~= 1 && j ~= column)),     vector = [l(i,j);r(i,j);lb(i,j);b(i,j);rb(i,j)]; end
            if (i == row && (j ~= 1 && j ~= column)),   vector = [lu(i,j);u(i,j);ru(i,j);l(i,j);r(i,j)]; end
            if (j == 1 && (i ~= 1 && i ~= row)),        vector = [u(i,j);ru(i,j);r(i,j);b(i,j);rb(i,j)]; end
            if (j == column && (i ~= 1 && i ~= row)),   vector = [lu(i,j);u(i,j);l(i,j);lb(i,j);b(i,j)]; end
        else
            if size == -4
                vector = [lu(i,j);ru(i,j);lb(i,j);rb(i,j)];
                if (i == 1 && j == 1),                      vector = rb(i,j); end
                if (i == 1 && j == column),                 vector = lb(i,j); end
                if (i == row && j == 1),                    vector = ru(i,j); end
                if (i == row && j == column),               vector = lu(i,j); end
                if (i == 1 && (j ~= 1 && j ~= column)),     vector = [lb(i,j);rb(i,j)]; end
                if (i == row && (j ~= 1 && j ~= column)),   vector = [lu(i,j);ru(i,j)]; end
                if (j == 1 && (i ~= 1 && i ~= row)),        vector = [ru(i,j);rb(i,j)]; end
                if (j == column && (i ~= 1 && i ~= row)),   vector = [lu(i,j);lb(i,j)]; end
            else %size == 4
                vector = [u(i,j);l(i,j);r(i,j);b(i,j)];
                if (i == 1 && j == 1),                      vector = [r(i,j);b(i,j)]; end
                if (i == 1 && j == column),                 vector = [l(i,j);b(i,j)]; end
                if (i == row && j == 1),                    vector = [u(i,j);r(i,j)]; end
                if (i == row && j == column),               vector = [u(i,j);l(i,j)]; end
                if (i == 1 && (j ~= 1 && j ~= column)),     vector = [l(i,j);r(i,j);b(i,j)]; end
                if (i == row && (j ~= 1 && j ~= column)),   vector = [u(i,j);l(i,j);r(i,j)]; end
                if (j == 1 && (i ~= 1 && i ~= row)),        vector = [u(i,j);r(i,j);b(i,j)]; end
                if (j == column && (i ~= 1 && i ~= row)),   vector = [u(i,j);l(i,j);b(i,j)]; end
            end
        end
       
        x(i,j) = sum(vector);
    end
end


end