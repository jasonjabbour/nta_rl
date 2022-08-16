function ShallowWater(ax_main, step, world_limit)

    %Amplitude
    A = 10;
    %Wave Length
    k_x = 100;
    k_y = 100;
    %Speed
    w = .1;
    %Perturbation
    phi = 1;
    %Resolution
    n = 20;
    
    %Create Grid of Water XY plane
    grid_range = linspace(-world_limit, world_limit, n);
    [x, y] = meshgrid(grid_range, grid_range);
    
    %Create Wave at this timestep
    z = A*sin(k_x*x + k_y*y+w*step+phi) - 10;

    %Plot Surface Wave
    surf_obj = surf(x, y, z, 'Parent', ax_main);
    drawnow

    %Color blue
    colormap([0.2 .2 .8])

    %Clear Wave to prepare for next step
    delete(surf_obj)
    

end