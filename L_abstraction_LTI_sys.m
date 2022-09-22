clear all
close all

A = [1,2;-1,1]/3;
l = 4;
alfa = norm(A,2);
sprintf('|A|_2 = %.3f', alfa);

%% Partition state space
part_num_x = 10;
part_num_y = 10;
tot_init_part = (part_num_x-1)*(part_num_y-1);

Ix = [-1,1];
Iy = [-1,1];
d_max = Ix(2);
d_min = (Ix(2)-Ix(1))/(part_num_x-1)/2;

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

%% Follow one set forward
l_partition = '1-5-5-5-2-2-2-2';

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
area_poly = zeros(1,l);

plot(mypoly)
[xc,yx] = centroid(mypoly);
trajectory(:,1) = [xc;yx];
init_area = area(mypoly);
area_poly(1) = area(mypoly)/init_area; 
for u = 2 : l
    mypoly.Vertices = (A*mypoly.Vertices')';
    [xc,yx] = centroid(mypoly);
    trajectory(:,u) = [xc;yx];
    area_poly(u) = area(mypoly)/init_area; 
    plot(mypoly)
end
plot(trajectory(1,:),trajectory(2,:), 'Color', 'red')
title(P_seq{q}{2})
figure
plot(area_poly)
title("Area l-partition")
%% Follow sets backwards
figure

H = ceil(log(d_min/d_max)/log(alfa));
Pre_diff_measure_traj = zeros(length_P_seq,H-1);
Union_measure_traj = zeros(length_P_seq,H);
Pre_measure_traj = zeros(length_P_seq,H);

k = linspace(0,H-1,H);

center = [0;0];
length = 2;
vertices = [center+1/2*length,center-1/2*length*[1;-1],center-1/2*length,center-1/2*length*[-1;1]];
mypoly = polyshape(vertices');
S_0 = cell(1,2);
index_0 = 0;
for i = 1 : length_P_seq
    if isinterior(P_seq{i}{1},0,0)
        index_0 = i;
        S_0 = P_seq{i};
    end
    set = P_seq{i}{1};
    union_set = set;
    union_set_l_seq = cell(H,1);
    set_l_seq = cell(H-1,1);
    for j = 1 : H
        Pre_measure_traj(i,j) = area(intersect(set,mypoly));
        Union_measure_traj(i,j) = area(union_set);
        set.Vertices = (A^(-1)*set.Vertices')';
        union_set = union(intersect(set,mypoly),union_set);
        union_set_l_seq{j} = P_seq{i}{2};
    end
    mu_inf = Union_measure_traj(i,H);
    Union_measure_traj(i,:) = Union_measure_traj(i,:)/mu_inf;
    Pre_measure_traj(i,:) = Pre_measure_traj(i,:)/mu_inf;
    Pre_diff_measure_traj(i,:) = diff(Union_measure_traj(i,:));
    % used to create the plot curve label
    for j = 1 : H-1        
        set_l_seq{j} = union_set_l_seq{j};
    end
    subplot(3,1,1)
    s = plot(k,Union_measure_traj(i,:));
    row = dataTipTextRow('l-seq',union_set_l_seq);
    s.DataTipTemplate.DataTipRows(end+1) = row;
    hold on
    subplot(3,1,2)
    s = plot(k,Pre_measure_traj(i,:));
    row = dataTipTextRow('l-seq',union_set_l_seq);
    s.DataTipTemplate.DataTipRows(end+1) = row;
    hold on
    subplot(3,1,3)
    s = plot(k(1:end-1),Pre_diff_measure_traj(i,:));
    row = dataTipTextRow('l-seq',set_l_seq);
    s.DataTipTemplate.DataTipRows(end+1) = row;
    hold on
end

subplot(3,1,1)
h = title('$\mu(\mathcal{D}\bigcap\bigcup_{i=0}^{H-1} Pre^i(S))/\mu_0^{\infty}(S)$','interpreter', 'latex');
h.FontSize = 15;
subplot(3,1,2)
h = title('$\mu(\mathcal{D}\bigcap Pre^i(S))/\mu_0^{\infty}(S)$','interpreter', 'latex');
h.FontSize = 15;
subplot(3,1,3)
h = title('$\mu(\mathcal{D}\bigcap Pre^{i+1}(S)\setminus Pre^i(S))/\mu_0^{\infty}(S)$','interpreter', 'latex');
h.FontSize = 15;

%% Lower bounding function

rho = abs(det(A)); %Choose
f = (1 - rho.^(-1))./(1-rho.^(k-H));
subplot(3,1,1)
plot(f,'Color','black','LineStyle','-.');
