% This script updates mesh element node numbering to ensure surface normals
% point in the same direction.

clc
clear all
close all
X = 1;
Y=2;
Z=3;
addpath(genpath(['../../../../cvt/cvt/matlab/third_party'])) % Automatically adds all subdirectories
% This script requires the geom3D functions to be added to the path
% (https://au.mathworks.com/matlabcentral/fileexchange/24484-geom3d)

v  = importdata('output/initial_nodes.txt');
f3d  = importdata('output/initial_elements.txt');
f3d = [f3d(:,5:8),f3d(:,1:4)];

f=[];

figure
axis equal

hold on
labels = arrayfun(@num2str,1:8, 'UniformOutput', false);
nodes = 1:8;
text(v(nodes,X),v(nodes,Y),v(nodes,Z),labels)
scatter3(v(:,X),v(:,Y),v(:,Z))

% Rearrange elements to have consistent surface normal
f = [f;f3d(:,1:4)];
f = [f;f3d(:,fliplr([1,2,6,5]))];
f = [f;f3d(:,fliplr([6,2,3,7]))];

drawMesh(v, f,'FaceAlpha',0.15);

n = faceNormal(v, f);
e = meshEdges(f);
drawFaceNormals(v, e, f)

checkMeshAdjacentFaces(v, e, f)

dlmwrite('output/final_elements.txt',f,'delimiter',',','precision',14)
dlmwrite('output/final_nodes.txt',v,'delimiter',',','precision',14)

exit