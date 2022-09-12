clear all
close all

A = [1,2;-1,1]/3;
l = 20;

%% Partition state space
part_num_x = 4;
part_num_y = 3;
tot_init_part = (part_num_x-1)*(part_num_y-1);

Ix = [-1,1];
Iy = [-1,1];

[Px,Py] = meshgrid(linspace(Ix(1), Ix(2), part_num_x), linspace(Iy(1), Iy(2), part_num_y));


Nodes = zeros(part_num_y,part_num_x,2);
Nodes(:,:,1) = Px;
Nodes(:,:,2) = Py;

%% Initialize partition
P_init = cell(tot_init_part,1);

for i = 1 : part_num_y-1
    for j = 1 : part_num_x-1
        vertices = [[Nodes(i,j,1); Nodes(i,j,2)],[Nodes(i+1,j,1); Nodes(i+1,j,2)],...
            [Nodes(i+1,j+1,1); Nodes(i+1,j+1,2)],[Nodes(i,j+1,1); Nodes(i,j+1,2)]];
        P_init{(i-1)*(part_num_x-1) + j} = {polyshape(vertices'),num2str((i-1)*(part_num_x-1) + j)};
    end
end
P_seq = P_init;
length_P_seq = length(P_seq);
close all

%% Construct l-abstraction
for v = 1 : l-1
    % Compute pre-image
    figure
    x0=10;
    y0=10;
    width=1500;
    height=700;
    set(gcf,'position',[x0,y0,width,height])
    subplot(1,2,1)
    grid on
    hold on
    for i = 1 : length_P_seq
        plot(P_seq{i}{1},'FaceColor','red','FaceAlpha',0.2)
        [xc,yc] = centroid(P_seq{i}{1});
        text(xc,yc, P_seq{i}{2}, 'HorizontalAlignment','center', 'VerticalAlignment','middle','Color','blue')
    end
    for i = 1 : length_P_seq
        P_seq{i} = {polyshape((inv(A)*(P_seq{i}{1}.Vertices)')'),P_seq{i}{2}};
        plot(P_seq{i}{1},'FaceColor','blue','FaceAlpha',0.2)
        [xc,yc] = centroid(P_seq{i}{1});
        text(xc,yc, P_seq{i}{2}, 'HorizontalAlignment','center', 'VerticalAlignment','middle')
    end
    title(sprintf('%i-Partition and Pre-image', v))
    axis equal
    % Find intersections
    length_P_seq = length(P_seq);
    P_intersections = cell(tot_init_part*length_P_seq,1);
    length_P_intersections = tot_init_part*length_P_seq;
    k = 0;
    for i = 1 : tot_init_part
        for j = 1 : length_P_seq
            polyout = intersect(P_init{i}{1},P_seq{j}{1});
            if polyout.NumRegions > 0
                k = k + 1;
                P_intersections{(i-1)*length_P_seq + j} = {polyout,append(P_init{i}{2},'-',P_seq{j}{2})};
            end
        end
    end
    sprintf("Number intersections %i",k)
    % Get l-partitions
    subplot(1,2,2)
    hold on
    P_seq = cell(k,1);
    length_P_seq = k;    
    for i = 1 : length_P_intersections
        if ~isempty(P_intersections{i})
            P_seq{k} = P_intersections{i};
            plot(P_seq{k}{1})
            [xc,yc] = centroid(P_seq{k}{1});
            text(xc,yc, P_seq{k}{2}, 'HorizontalAlignment','center', 'VerticalAlignment','middle')
            k = k - 1;    
        end
    end
    title(sprintf('%i-Partition', v+1))
    axis equal
end

%% Follow One set
l_partition = '8-7-12-12-13';

for q = 1 : length_P_seq
    if strcmp(P_seq{q}{2},l_partition)
        display("l-partition found");
        break
    end
end
figure
hold on

for i = 1 : part_num_y-1
    for j = 1 : part_num_x-1
        vertices = [[Nodes(i,j,1); Nodes(i,j,2)],[Nodes(i+1,j,1); Nodes(i+1,j,2)],...
            [Nodes(i+1,j+1,1); Nodes(i+1,j+1,2)],[Nodes(i,j+1,1); Nodes(i,j+1,2)]];
        P_init{(i-1)*(part_num_x-1) + j} = {polyshape(vertices'),num2str((i-1)*(part_num_x-1) + j)};
        plot(P_init{(i-1)*(part_num_x-1) + j}{1},'FaceColor','red','FaceAlpha',0.2)
        [xc,yc] = centroid(P_init{(i-1)*(part_num_x-1) + j}{1});
        text(xc,yc, P_init{(i-1)*(part_num_x-1) + j}{2}, 'HorizontalAlignment','center', 'VerticalAlignment','middle','Color','blue')
    end
end

mypoly = P_seq{q}{1};
trajectory = zeros(2,l);

plot(mypoly)
[xc,yx] = centroid(mypoly);
trajectory(:,1) = [xc;yx];
for u = 2 : l
    mypoly.Vertices = (A*mypoly.Vertices')';
    [xc,yx] = centroid(mypoly);
    trajectory(:,u) = [xc;yx];
    plot(mypoly)
end
plot(trajectory(1,:),trajectory(2,:), 'Color', 'red')
title(P_seq{q}{2})