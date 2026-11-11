clear

%%%%%%%%%%%%% Scenario Parameters %%%%%%%%%%%%%
startTime = datetime(2026,09,22,0,0,0);
stopTime = startTime + seconds(4500);

sampleTime = 60;

altitude = 781e3; %m
inclination = 86.4; %deg
Sat_count = 66;
Constellation_planes = 6;

name = ["Redu, Belgium"];
lat = [50];
lon = [5.15];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Spacecraft  %%%%%%%%%
sc = satelliteScenario(startTime,stopTime,sampleTime);

sat = walkerStar(sc, altitude+6378.14e3, inclination, Sat_count, Constellation_planes, 2, Name="Iridium");

% sat = walkerDelta(sc,  500e3+6378.14e3, 53, 72, 12, 2, Name="Starlink");

Groundstations = groundStation(sc, "Name", name, "Latitude", lat,"Longitude", lon);

% satelliteScenarioViewer(sc);

access_ob = access(sat, Groundstations);
% accessintervals = accessIntervals(access_ob);

% Creating Network
% Creating all interplane access and cross plane access for
% 1-2-3-4-5-6 but not 6-1 (because of the opposite realtive motion btween
% these planes)
%%%%%%%%% Aircraft  %%%%%%%%%
%% Aircraft trajectory: Ottawa -> Quebec City -> Halifax at cruise altitude
%            lat      lon      alt [m]
waypoints = [45.32   -75.67   11000;   % Ottawa (YOW)
             46.79   -71.39   11000;   % Quebec City (YQB)
             44.88   -63.51   11000];  % Halifax (YHZ)
toa = [0; 1500; 4500];                 % time of arrival at each waypoint [s]
traj = geoTrajectory(waypoints, toa, AutoPitch = true, AutoBank = true);
 
% Add it to the scenario as a moving platform
ac = platform(sc, traj, Name = "AC1");

%%%%%%%%% Access  %%%%%%%%%
network_accesses = [];

grouped_idx = [];
range_length = Sat_count/Constellation_planes;
for current_plane = 1:Constellation_planes
    next_plane = current_plane + 1;
    if next_plane > Constellation_planes
        next_plane = 1;
    end
    end_idx = current_plane*range_length;
    start_idx = end_idx - (range_length-1);
    grouped_idx = [grouped_idx; start_idx, end_idx;];
    for j = start_idx:end_idx
        current_idx = j;
        next_idx = j+1;
        if next_idx > end_idx
            next_idx = start_idx;
        end
        current_orbital_elms = orbitalElements(sat(current_idx));
        current_true_anomaly = current_orbital_elms.TrueAnomaly;
        next_plane_end_idx = next_plane*range_length;
        next_plane_start_idx = next_plane_end_idx - (range_length-1);
        for next_plane_idx = next_plane_start_idx:next_plane_end_idx
            next_plane_orbital_elms = orbitalElements(sat(next_plane_idx));
            next_plane_true_anomaly = next_plane_orbital_elms.TrueAnomaly;
            true_anomaly_diff = abs(current_true_anomaly-next_plane_true_anomaly);
            [ az, el, rng] = aer(sat(current_idx),sat(next_plane_idx)); 
            if exist('closest_true_anomaly', 'var')
                if true_anomaly_diff < closest_true_anomaly
                    closest_true_anomaly = true_anomaly_diff;
                    closest_true_anomaly_idx = next_plane_idx;
                else
                end
            else
                closest_true_anomaly = true_anomaly_diff;
                closest_true_anomaly_idx = next_plane_idx;
            end 
        end
        clear closest_true_anomaly
        clear closest_rng

        interplane_access = access(sat(current_idx), sat(next_idx));
        network_accesses = [network_accesses; accessIntervals(interplane_access)];
        if current_plane == Constellation_planes
            continue
        else
            crossplane_access = access(sat(current_idx), sat(closest_true_anomaly_idx));
            network_accesses = [network_accesses; accessIntervals(crossplane_access)];
        end 
    end
end

aircraft_access = access(ac, sat);
show(ac.GroundTrack);

% satelliteScenarioViewer(sc);
