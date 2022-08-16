%% --Digital Twin Class --
classdef DigitalTwin

    %% -- Public Properties --
    properties
        MainFigure;
        MainAxis; 
        DigitalTwinGeometryPlot; 
        DigitalTwinGeometry;
        GeometryFile = "models/USS_Conrad.stl";
        Set_xlim = 200;
        Set_ylim = 200;
        Set_zlim = 200;
        DigitalTwinInitDirection = [0 0 1];
        DigitalTwinInitAngle = 180; 

    end


    %% Class Methods
    methods

        % -- Constructor --
        function obj = DigitalTwin()
            
            %Create Main Figure
            obj.MainFigure = figure(1);
            obj.MainAxis = axes('Parent', obj.MainFigure);

            %Load Digital Twin Geometry
            [obj.DigitalTwinGeometryPlot, obj.DigitalTwinGeometry] = obj.LoadModelGeometry();

            %Set initial Orientation of Digital Twin
            rotate(obj.DigitalTwinGeometryPlot, obj.DigitalTwinInitDirection, obj.DigitalTwinInitAngle);

            %Set limits of axis
            obj.SetAxisBounds(obj.MainAxis);

        end

        % -- Setter Methods --
        function obj = set.Set_xlim(obj, xlim)
            obj.Set_xlim = xlim; 
        end

        function obj = set.Set_ylim(obj, ylim)
            obj.Set_ylim = ylim; 
        end

        function obj = set.Set_zlim(obj, zlim)
            obj.Set_zlim = zlim; 
        end

        % -- Getter Methods --
        function xlim = get.Set_xlim(obj)
            xlim = obj.Set_xlim;
        end

        function ylim = get.Set_ylim(obj)
            ylim = obj.Set_ylim;
        end

        function zlim = get.Set_zlim(obj)
            zlim = obj.Set_zlim;
        end

        function main_figure = get.MainFigure(obj)
            main_figure = obj.MainFigure; 
        end
     
        function main_axis = get.MainAxis(obj)
            main_axis = obj.MainAxis; 
        end

        function gm_plt = get.DigitalTwinGeometryPlot(obj)
            gm_plt = obj.DigitalTwinGeometryPlot; 
        end

        function gm = get.DigitalTwinGeometry(obj)
            gm = obj.DigitalTwinGeometry; 
        end

        %Load Model Geometry: Return Geometry and Plot
        function [gm_plt, gm] = LoadModelGeometry(obj)
            gm = importGeometry(obj.GeometryFile);
            gm_plt = pdegplot(gm);
        end
        
        %Set the Axis Limits given an Axis Object
        function SetAxisBounds(obj, axis)
            xlim(axis, [-obj.Set_xlim obj.Set_xlim]);
            ylim(axis, [-obj.Set_ylim obj.Set_ylim]);
            zlim(axis, [-obj.Set_zlim obj.Set_zlim]);
        end

    end

end
