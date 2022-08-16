%Import Variables
config_variables = ship_config();

%Set correct Names
set_xlim = config_variables.set_xlim;
set_ylim = config_variables.set_ylim;
set_zlim = config_variables.set_zlim;

%Declare Figure
figure_main = figure(1);
ax_main = axes('Parent',figure_main);

%Insert Ship 3D Model
gm = importGeometry("models/USS_Conrad.stl");
threeD_ship = pdegplot(gm);

%Plot Limits
xlim([-set_xlim set_xlim]);
ylim([-set_ylim set_ylim]);
zlim([-set_zlim set_zlim]);

%Set Base Position
init_direction = [0 0 1]; 
init_angle = 180;
rotate(threeD_ship, init_direction, init_angle);

%To rotate ship about x axis (roll)
direction = [1 0 0]; 

%Set time
t = 3000; 
change_angle = 1;

for step = 1:t

    %Change Direction
    if mod(step, 20) == 0
        change_angle=-1*change_angle;
    end
    angle = .5*change_angle;
    
    %Rotate 3D Ship Model
    rotate(threeD_ship, direction, angle)
    hold on

    %Step Waves
    ShallowWater(ax_main, step, set_xlim);

end 
