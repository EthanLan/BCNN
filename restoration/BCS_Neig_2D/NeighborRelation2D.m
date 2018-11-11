function Indi_Neig = NeighborRelation2D(Indi_Data,size)

len = length(Indi_Data);
Indi_Neig = zeros(len,1);

row = sqrt(len./3);
column = row;
sub_len = row * column;
sub_index = 1:sub_len;

for k = 1:3 % 3 subbands
	idx = sub_index + (k-1) * sub_len;
	data = Indi_Data(idx);data = reshape(data,row,column);
    
	neighbour = neighbor_process1(data,row,column,size);
	IdiNeighbour = reshape(neighbour,sub_len,1);
    Indi_Neig(idx) = IdiNeighbour;
end